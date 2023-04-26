import os
import numpy as np
from collections import OrderedDict
import torch
from torch import Tensor
import cv2
from . import networks


class Pix2PixModel:
    def __init__(self, opt, val_dataloader):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.backends.cudnn.benchmark = True
        self.image_paths = []
        self.metric = 0

        self.val_dataloader = val_dataloader
        self.output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.netG = networks.init_net(networks.UNet(opt.input_nc, opt.output_nc, 8, opt.ngf, not opt.no_dropout).to(self.device))

        if self.isTrain:
            self.netD = networks.init_net(networks.PatchGAN(opt.input_nc + opt.output_nc, opt.ndf).to(self.device))
            self.criterionGAN = networks.define_loss(opt.gan_mode, self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_D, self.optimizer_G]
            self.models = [(self.netD, "discriminator"), (self.netG, "generator")]
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            self.optimizers = []
            self.models = [(self.netG, "generator")]

        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f"learning rate {old_lr:.7f} -> {lr:.7f}")

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for (net, name) in self.models:
            save_filename = f"{epoch}_net_{name}.pth"
            save_path = os.path.join(self.save_dir, save_filename)
            torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch):
        for (net, name) in self.models:
            load_filename = f"{epoch}_net_{name}.pth"
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print(f"loading the model from {load_path}")
            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def update_discriminator(self, fake_img: Tensor, input_img: Tensor, target_img: Tensor) -> None:
        for param in self.netD.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()

        fake_input = torch.cat((input_img, fake_img), 1)
        pred_fake = self.netD(fake_input.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_input = torch.cat((input_img, target_img), 1)
        pred_real = self.netD(real_input)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

    def update_generator(self, fake_img: Tensor, input_img: Tensor, target_img: Tensor) -> None:
        for param in self.netD.parameters():
            param.requires_grad = False
        self.optimizer_G.zero_grad()

        fake_input = torch.cat((input_img, fake_img), 1)
        pred_fake = self.netD(fake_input)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(fake_img, target_img) * 100

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_discriminator(self.fake_B, self.real_A, self.real_B)
        self.update_generator(self.fake_B, self.real_A, self.real_B)

    def generate_and_save(self, epoch: int) -> None:
        was_training = self.netG.training
        self.netG.eval()
        with torch.no_grad():
            for (i, data) in enumerate(self.val_dataloader):
                input = data['A'].to(self.device)
                target = data['B'].to(self.device)

                pred = self.netG(input).cpu().numpy()
                pred = np.squeeze(pred)
                pred = pred * 0.5 + 0.5
                pred = np.clip(pred, 0, 1)
                pred = np.transpose(pred, [1, 2, 0])

                input = input.cpu().numpy()
                input = np.squeeze(input)
                input = input * 0.5 + 0.5
                input = np.transpose(input, [1, 2, 0])

                target = target.cpu().numpy()
                target = np.squeeze(target)
                target = target * 0.5 + 0.5
                target = np.transpose(target, [1, 2, 0])

                error = np.abs(pred - target)

                output1 = np.concatenate((input, target), axis=1)
                output2 = np.concatenate((pred, error), axis=1)
                output = np.concatenate((output1, output2), axis=0)
                output = (output * (2**16-1)).astype(np.uint16)
                output = np.concatenate((output[:, :, 2:3], output[:, :, 1:2], output[:, :, 0:1]), axis=2)
                if os.path.exists(self.output_dir) == False:
                    os.mkdir(self.output_dir)
                cv2.imwrite(f"{self.output_dir}/epoch_{epoch+1}_{i+1}.png", output)
        self.netG.train(was_training)

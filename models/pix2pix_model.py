import os
import numpy as np
import torch
from torch import Tensor
import cv2
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    and a '--gan_mode' BCE GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt, val_dataloader):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.val_dataloader = val_dataloader
        self.output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.init_net(networks.UNet(opt.input_nc, opt.output_nc, 8, opt.ngf, not opt.no_dropout).to(self.device))

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.init_net(networks.PatchGAN(opt.input_nc + opt.output_nc, opt.ndf).to(self.device))

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.define_loss(opt.gan_mode, self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

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

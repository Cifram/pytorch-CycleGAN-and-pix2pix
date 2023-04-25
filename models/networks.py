import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net):
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    print('initialize network')
    net.apply(init_weights)  # apply the initialization function <init_func>


def init_net(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net)
    return net


def define_G(input_nc, output_nc, ngf, netG, use_dropout=False, gpu_ids=[]):
    if netG == 'unet_128':
        net = UNet(input_nc, output_nc, 7, ngf, dropout=use_dropout)
    elif netG == 'unet_256':
        net = UNet(input_nc, output_nc, 8, ngf, dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, gpu_ids)


def define_D(input_nc, ndf, gpu_ids=[]):
    net = NLayerDiscriminator(input_nc, ndf, layers=3)
    return init_net(net, gpu_ids)


def define_loss(gan_mode: str, device):
    if gan_mode == 'bce':
        return BasicGanLoss(device, nn.BCEWithLogitsLoss())
    elif gan_mode == 'mse':
        return BasicGanLoss(device, nn.MSELoss())
    elif gan_mode == 'wgan':
        return WGanLoss(device)
    else:
        raise ValueError(f'Invalid loss type: {gan_mode}')


##############################################################################
# Classes
##############################################################################

class BasicGanLoss:
    def __init__(self, device, loss_fn: nn.Module):
        self.device = device
        self.loss_fn = loss_fn

    def __call__(self, pred: Tensor, target_is_real: bool) -> Tensor:
        if target_is_real:
            target_tensor = torch.ones_like(pred)
        else:
            target_tensor = torch.zeros_like(pred)
        target_tensor = target_tensor.to(self.device)
        return self.loss_fn(pred, target_tensor)

    def post_process_discriminator(self, discriminator: nn.Module) -> None:
        pass


class WGanLoss:
    def __init__(self, device):
        self.device = device

    def __call__(self, pred: Tensor, target_is_real: bool) -> Tensor:
        if target_is_real:
            loss = -pred.mean()
        else:
            loss = pred.mean()
        return loss

    def post_process_discriminator(self, discriminator: nn.Module) -> None:
        with torch.no_grad():
            for param in discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, start_filters: int, dropout: bool = True) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, start_filters, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='replicate'),
            InnerUNet(start_filters, start_filters*8, depth-1, dropout),
            UpConv(start_filters*2, out_channels, normalize=False, dropout=False, use_bias=True),
            nn.Tanh(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


class UpConv(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True, dropout: bool = True, use_bias: bool = False) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_filters, out_filters, kernel_size=5, stride=1, padding=2, bias=use_bias, padding_mode='replicate'),
        )
        if normalize:
            self.model.append(nn.BatchNorm2d(out_filters))
        if dropout:
            self.model.append(nn.Dropout(0.5))

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


class DownConv(nn.Module):
    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False, padding_mode='replicate'),
        )
        if normalize:
            self.model.append(nn.BatchNorm2d(out_filters))

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)


class InnerUNet(nn.Module):
    def __init__(self, filters: int, max_filters: int, depth: int, dropout: bool) -> None:
        super().__init__()
        inner_filters = min(filters*2, max_filters)
        if depth == 1:
            self.model = nn.Sequential(
                DownConv(filters, inner_filters, normalize=False),
                UpConv(inner_filters, filters, dropout=False),
            )
        else:
            self.model = nn.Sequential(
                DownConv(filters, inner_filters),
                InnerUNet(inner_filters, max_filters, depth-1, dropout=dropout),
                UpConv(inner_filters*2, filters, dropout=dropout and inner_filters == max_filters),
            )

    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([input, self.model(input)], 1)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels, start_filters=64, layers=3):
        super(NLayerDiscriminator, self).__init__()
        sequence = [nn.Conv2d(input_channels, start_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        filters = start_filters
        max_filters = start_filters * 8
        for n in range(0, layers):  # gradually increase the number of filters
            next_filters = min(filters * 2, max_filters)
            final = n == layers - 1
            sequence += [
                nn.Conv2d(filters, next_filters, kernel_size=4, stride=1 if final else 2, padding=1, bias=False),
                nn.BatchNorm2d(next_filters),
                nn.LeakyReLU(0.2, True)
            ]
            filters = next_filters

        sequence += [nn.Conv2d(filters, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

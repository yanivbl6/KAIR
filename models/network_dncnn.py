import torch
import torch.nn as nn
import models.basicblock as B


"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""

def norm_act(output_net, new_norm, epsilon = 1e-5):
    output_net_shape = output_net.shape
    output_net = output_net.view(output_net.size(0), -1)
    frac = new_norm / (torch.norm(output_net, dim=1) + epsilon)
    output_net = output_net * (1 - torch.nn.functional.relu(1 - frac)).unsqueeze(1)
    return output_net.view(output_net_shape)

def calc_norm(data, gray_scale=False):
    if gray_scale:
        data_norm = data.norm(dim=(2, 3), keepdim = True)
    else:
        data_norm = torch.norm(data.norm(dim=(2, 3), keepdim = True), dim=1, keepdim = True)
    return data_norm

class NormAct(nn.Module):
    def __init__(self):
        super(NormAct, self).__init__()  # init the base class

    def forward(self, output_net, new_norm):
        if new_norm is None:
            return output_net
        else:
            return norm_act(output_net, new_norm)


# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR', normact = None, use_real_norm = False, constant_norm = True):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)
        self.normact = NormAct()
        self.new_norm = normact
        self.sigma = None
        self.in_nc = in_nc
        self.use_real_norm = use_real_norm

        self.constant_norm = constant_norm

    def feed_sigma(self, sigma):
        self.sigma = sigma

    def feed_ynorm(self,ynorm):
        self.ynorm = ynorm

    def feed_xnorm(self,xnorm):
        self.xnorm = xnorm

    def forward(self, x):
        n = self.model(x)
        if self.constant_norm:
            if self.training:
                return self.normact(x-n,  self.new_norm)
            else:
                if self.new_norm is None:
                    return self.normact(x-n,  self.new_norm)
                else:
                    gray_scale = self.in_nc == 1
                    d = x.shape[2] * x.shape[3] * self.in_nc
                    if self.use_real_norm:
                        norm_frac = (self.xnorm/self.new_norm).to(device = x.device)
                    else:
                        norm_frac = (torch.sqrt(self.ynorm**2 - d * ((self.sigma/255.0)**2))/self.new_norm).to(device = x.device)
                    return norm_frac*self.normact(x-n,  self.new_norm)
        else:
            gray_scale = self.in_nc == 1
            d = x.shape[2] * x.shape[3] * self.in_nc
            if self.use_real_norm:
                norm_frac = (self.xnorm).to(device = x.device)
            else:
                norm_frac = (torch.sqrt(self.ynorm**2 - d * ((self.sigma/255.0)**2))).to(device = x.device)
            return norm_frac*self.normact(x-n,  1.0)
# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L =[]
        L.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = B.sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x-n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    from utils import utils_model
    import torch
    model1 = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR')
    print(utils_model.describe_model(model1))

    model2 = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    print(utils_model.describe_model(model2))

    x = torch.randn((1, 1, 240, 240))
    x1 = model1(x)
    print(x1.shape)

    x = torch.randn((1, 2, 240, 240))
    x2 = model2(x)
    print(x2.shape)

    #  run models/network_dncnn.py

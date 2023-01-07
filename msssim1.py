# Computing MS-SSIM (Multi-Scale Structural Similarity) scores
# The code is borrowed from the following github repository:
#   -   https://github.com/jorge-pessoa/pytorch-msssim
#   -   https://ece.uwaterloo.ca/~z70wang/research/ssim/

import torch
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss=[exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]
    gauss_tensor = torch.Tensor(gauss)
    gaussian_ = gauss_tensor/gauss_tensor.sum()
    return gaussian_


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
def gaussian_filter(img,window,channel):
    result = F.conv2d(img, window, groups=channel)
    return result

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):

    max_val = 255 if torch.max(img1)>128 else 1
    min_val = -1 if torch.min(img1)<-0.5 else 0
    L = max_val - min_val

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = gaussian_filter(img1,window,channel)
    mu1_sq = pow(mu1,2)
    mu2 = gaussian_filter(img2,window,channel)
    mu2_sq = pow(mu2,2)
    mu1_mu2 = mu1 * mu2

    sigma1_ori = gaussian_filter(img1*img1 ,windoe,channel)
    sigma1_sq = sigma1_ori - mu1_sq

    sigma2_ori = gaussian_filter(img2 * img2, windoe, channel)
    sigma2_sq = sigma2_ori - mu2_sq

    sigma12_ori = gaussian_filter(img1 * img2, windoe, channel)
    sigma12 = sigma12_ori - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ret = ssim_map.mean() if size_average==True else ssim_map.mean(1).mean(1).mean(1)
    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ret,cs
    return ret

def cal_mssim_mcs(img1,img2,levels, window_size=window_size, size_average=size_average):
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    return mssim, mcs
def msssim(img1, img2, window_size=11, size_average=True):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim,mcs = cal_mssim_mcs(img1,img2, levels,window_size=window_size, size_average=size_average)
    output = torch.prod( (mcs ** weights)[:-1]) * (mssim ** weights)[-1]
    return output


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

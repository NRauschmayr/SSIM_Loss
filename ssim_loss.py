from math import exp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.loss import Loss

# Implementation is taken from https://github.com/Po-Hsun-Su/pytorch-ssim
# and translated into MXNet Gluon

def gaussian(size, sigma):
    gauss = mx.nd.array([exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)], mx.cpu())
    return gauss/gauss.sum()

def create_window(size, channel):
    _1D_window = gaussian(size, 1.5).expand_dims(1)
    _2D_window = (_1D_window * _1D_window.T).expand_dims(0).expand_dims(0)
    window = mx.nd.array(_2D_window.broadcast_to([channel, 1, size, size]))
    window.attach_grad()
    return window

class SSIM(Loss):
    def __init__(self, size = 11, weight=None, batch_axis=0, **kwargs):
        super(SSIM, self).__init__(weight, batch_axis, **kwargs)
        self.size = size
        self.channel = 3
        self.window = create_window(size, self.channel)
        
    def hybrid_forward(self, F, img1, img2):
        self.window = self.window.as_in_context(img1.context)
        mu1 = F.Convolution(data=img1, weight=self.window, kernel=(self.size,self.size), no_bias=True, pad=(self.size//2,self.size//2), num_filter=self.channel, num_group = self.channel)
        mu2 = F.Convolution(data=img2, weight=self.window, kernel=(self.size,self.size), no_bias=True, pad=(self.size//2,self.size//2), num_filter=self.channel, num_group = self.channel)
        
        mu1_sq = F.power(mu1,2)
        mu2_sq = F.power(mu2,2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.Convolution(img1*img1, weight=self.window, kernel=(self.size,self.size), no_bias=True, pad=(self.size//2,self.size//2), num_filter=self.channel, num_group = self.channel) - mu1_sq
        sigma2_sq = F.Convolution(img2*img2, weight=self.window, kernel=(self.size,self.size), no_bias=True, pad=(self.size//2,self.size//2), num_filter=self.channel, num_group = self.channel) - mu2_sq
        sigma12 = F.Convolution(img1*img2, weight=self.window, kernel=(self.size,self.size), no_bias=True, pad=(self.size//2,self.size//2), num_filter=self.channel, num_group = self.channel) - mu1_mu2
       

        ssim_map = ((2*mu1_mu2 + 0.0001)*(2*sigma12 + 2.7e-08))/((mu1_sq + mu2_sq + 0.0001)*(sigma1_sq + sigma2_sq + 2.7e-08))

        return ssim_map.mean()

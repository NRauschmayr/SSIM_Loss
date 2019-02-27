import cv2
from matplotlib import pyplot as plt
from matplotlib import colors
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import numpy as np
import sys
import ssim_loss

img1 = cv2.imread(sys.argv[1])
img1 = cv2.resize(img1, (200,200))
img1 = mx.nd.array(np.transpose(img1, (2,0,1))/255.0)
img1 = img1.expand_dims(axis=0)
img2 = mx.nd.random.uniform(0,1,shape=img1.shape)
img2.attach_grad()

img3 = mx.nd.random.uniform(0,1,shape=img1.shape)
img3.attach_grad()

ssim_loss = ssim_loss.SSIM()
l2_loss   = gluon.loss.L2Loss()

mean2 = mx.nd.zeros(img1.shape)
var2  = mx.nd.zeros(img1.shape)

mean3 = mx.nd.zeros(img1.shape)
var3  = mx.nd.zeros(img1.shape)

counter = 0
while counter < 100:
  counter = counter + 1
  with autograd.record():
    loss2 = -ssim_loss(img1, img2)
    
  loss2.backward()
  mx.nd.adam_update(img2, img2.grad, mean2, var2, out=img2, lr=0.01)

  with autograd.record():
    loss3 = l2_loss(img1, img3)
  loss3.backward()
  mx.nd.adam_update(img3, img3.grad, mean3, var3, out=img3, lr=0.01)
  
  
  fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(ncols=5, figsize=(20, 5)) 
  ax0.axis("off")
  ax1.axis("off")
  ax2.axis("off")
  ax3.axis("off")
  ax4.axis("off")
  image1 = np.transpose(img1[0,:,:,:].asnumpy(), (1,2,0))
  image2 = np.transpose(img2[0,:,:,:].asnumpy(), (1,2,0))
  image3 = np.transpose(img3[0,:,:,:].asnumpy(), (1,2,0))
  
  ax0.imshow(image1)
  ax1.imshow(image2)
  ax1.set_title("SSIM Loss: " + str(-loss2.asscalar()), fontsize=12)
  ax2.imshow(np.abs(image2 - image1))
  ax2.set_title("Difference",fontsize=12)
  ax3.imshow(image3, interpolation='nearest')
  ax3.set_title("L2 Loss: " + str(loss3.asscalar()), fontsize=12)
  ax4.imshow(np.abs(image3 - image1))  
  ax4.set_title("Difference", fontsize=12)     
  plt.pause(0.05)
plt.show()





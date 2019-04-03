from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import numpy
import time

from visdom import Visdom
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn as nn
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib
import torchvision.utils as vutils
from torchvision import transforms, datasets

data_dir = 'data/test_set'
img_size=256
batch_size=4

data_transform=transforms.Compose([
    transforms.Scale(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])


dataset=datasets.ImageFolder(data_dir,data_transform)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=2)

real_batch=next(iter(dataloader))



def visualize_batch(batch,trans=True,nrow=4,save=False,name=None):
    grids = vutils.make_grid(batch[:], padding=2, normalize=True, nrow=nrow,range=(batch.min().item(),batch.max().item()))
    if trans:
        grids = np.transpose(grids.detach().cpu().numpy(), (1, 2, 0))
    plt.figure()
    plt.imshow(grids)
    if save:
        path='pic/'+name+'.png'
        plt.savefig(path)
    else:
       plt.show()



# for i in range(3):
    # visualize_batch(real_batch[0])
    # real_batch=iter(dataloader).next()
    # time.sleep(2)


for i, image in enumerate(dataloader):
    print(image[1])




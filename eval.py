
import torch
import numpy as np
import PIL
import XuNet
import model
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from PIL import Image
import cv2
from torchvision import transforms, datasets
import pytorch_ssim
import pytorch_msssim
import argparse

######################################################################

#make a parser
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default=-1,type=int)
parser.add_argument('--epoch',default=70,type=int)   #the stratpoint means the currently epoch,and load the last epoch data
parser.add_argument('--stegos',default=False,type=bool)
parser.add_argument('--reveals',default=False,type=bool)
parser.add_argument('--secrets',default=True,type=bool)


params=parser.parse_args()





epoch=params.epoch
make_stego=params.stegos
make_reveals=params.reveals
make_secrets=params.secrets
# threseld=0.9
one_channel='data/one'
data_dir = 'data/test_set'
model_dir= 'model/'
stego_dir='stegos/'
reveal_dir='reveals/'



batch_size=4
img_size = 256

# alpha=0.5
# beta=0.3
# gamma=0.85

gpu=params.gpu
gpu_available =  True if gpu>=0 else False


device=torch.device("cuda:%d"%(gpu) if gpu_available else "cpu")


data_transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])


dataset=datasets.ImageFolder(data_dir,data_transform)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1)

dataset_one=datasets.ImageFolder(one_channel,data_transform)

dataloader_one=torch.utils.data.DataLoader(dataset_one,batch_size=batch_size,shuffle=False,num_workers=1)


#initialize the model and load the params

encoder=model.Encoder()
encoder=encoder.to(device)

#decoder (discriminator)
decoder=model.Decoder()
decoder=decoder.to(device)

#steganalyzer
# steganalyzer = XuNet.Steganalyzer()
# steganalyzer=steganalyzer.to(device)


ssim_loss=pytorch_ssim.SSIM()
mssim_loss=pytorch_msssim.MSSSIM()
mse_loss=nn.MSELoss()
#dis_loss=nn.BCELoss()


print('loading params')

path = model_dir + '/' + str(epoch) + '.pth.tar'    #load theepoch params


checkpoint=torch.load(path,map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['deocoder_state_dict'])
# steganalyzer.load_state_dict(checkpoint['stganalyzer_state_dict'])

encoder.eval()
decoder.eval()
# steganalyzer.eval()

cover_ssim=[]
cover_mse=[]
cover_mssim=[]
secret_ssim=[]
secret_mse=[]
secret_mssim=[]

stego_step=0
reveal_step=0
secret_step=0


for i,data in enumerate(zip(dataloader,dataloader_one)):

        images=data[0][0]
        ones=data[1][0]

        if len(images) != batch_size: break
        covers=images
        secrets=ones
      #  print(mse_loss(covers,secrets).item())
        secrets = 0 + 0.299 * secrets[:, 0, :, :] + 0.587 * secrets[:, 1, :, :] + 0.114 * secrets[:, 2, :, :]
        secrets=secrets.view(-1,1,256,256)
        #visualize_batch(secrets)
        #print(covers.shape,secrets.shape)
        #transfer it to device
        covers=covers.to(device)
        secrets=secrets.to(device)

        #feed in the network
        # steganalyzer.zero_grad()
        # encoder.zero_grad()
        # decoder.zero_grad()

        stegos=encoder(covers,secrets)

        reveals=decoder(stegos)

        #forward finish

        s_mse = mse_loss(reveals, secrets)
        c_mse = mse_loss(stegos, covers)

        cover_mse.append(c_mse.item())
        secret_mse.append(s_mse.item())

        s_ssim = ssim_loss(reveals, secrets)
        c_ssim = ssim_loss(covers, stegos)

        cover_ssim.append(c_ssim.item())
        secret_ssim.append(s_ssim.item())

        s_mssim = mssim_loss(secrets, reveals)
        c_mssim = mssim_loss(covers, stegos)

        cover_mssim.append(c_mssim.item())
        secret_mssim.append(s_mssim.item())


        if make_stego:
            stego_output=stegos.permute(0,2,3,1).detach().numpy()*255

            for image in stego_output:
                path="stegos/"+str(stego_step+1)+".jpg"
                r=np.expand_dims(image[:,:,0],axis=2)
                g=np.expand_dims(image[:,:,1],axis=2)
                b=np.expand_dims(image[:,:,2],axis=2)

                out=np.concatenate((b,g,r),axis=2)

                cv2.imwrite(path,out)
                stego_step+=1
                

        if make_reveals:
            reveals_output=reveals.view(-1,256,256,1).detach().numpy()*255

            for image in reveals_output:
                path="reveals/"+str(reveal_step+1)+".jpg"


                cv2.imwrite(path,image)
                reveal_step+=1

        if make_secrets:
            onec_secret=secrets.view(-1,256,256,1).detach().numpy()*255
            for image in onec_secret:
                path="secrets/"+str(secret_step+1)+".jpg"


                cv2.imwrite(path,image)
                secret_step+=1




mean1=np.mean(cover_ssim)
mean2=np.mean(secret_ssim)
mean3=np.mean(cover_mse)
mean4=np.mean(secret_mse)






print(' ssim: %.4f|%.4f || mse: %.4f|%.4f'
                  % ( mean1, mean2, mean3, mean4))


print('finish')
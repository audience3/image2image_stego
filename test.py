'''
training script
made by audience on Dec 11,2018

'''

import torch
import numpy as np
import argparse
import XuNet
import model
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim

from torchvision import transforms, datasets
import pytorch_ssim
import pytorch_msssim
from visdom import Visdom

######################################################################

#make a parser
parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default=-1,type=int)
parser.add_argument('--start',default=40,type=int)   #the stratpoint means the currently epoch,and load the last epoch data
parser.add_argument('--resume',default=False,type=bool)

params=parser.parse_args()




######################################################################
manual_seed = 999
global_step=0

# data_dir = 'data/train_set'
data_dir='data/train_set'
model_dir= 'model'
# hyperparas definition
batch_size=8
img_size = 256
learning_rate=0.0001
weight_decay=1e-5
epochs=80
gpu=params.gpu
gpu_available =  True if gpu>=0 else False
resume=params.resume

######################################################################
#visdom visualization
viz=Visdom()
assert viz.check_connection()
initial_c=torch.rand(batch_size,3,img_size,img_size)
initial_s=torch.rand(batch_size,1,img_size,img_size)
temp=np.linspace(1,5,5)
plt.plot(temp)


result_c=viz.images(initial_c,nrow=batch_size//2,opts=dict(caption="cover vs. stego"))
result_s=viz.images(initial_s,nrow=batch_size//2,opts=dict(caption="secret vs. reveal"))

lc_ssim=viz.matplot(plt)
ls_ssim=viz.matplot(plt)
l_dis=viz.matplot(plt)
l_net=viz.matplot(plt)






# loss hyper
#L(c, c0) = α (1-SSIM(c, c0)) +(1-α)(1 MSSIM(c,c0))+ β MSE(c,c0)
#L(s, s0) = α (1-SSIM(s, s0)) +(1-α)(1 MSSIM(s,s0))+ β MSE(s,s0)
#L(c, c0, s, s0) = L(c, c0) + λ L(s, s0)

alpha=0.6
beta=0.5
gamma=0.85
lambda0=0.002
# optimizer hyper for adam

beta1=0.9
beta2=0.999


#something visualization

def visualize_batch(batch,trans=True,nrow=4,save=False,name=None):
    grids = vutils.make_grid(batch.to(device)[:], padding=2, normalize=True, nrow=nrow,range=(batch.min().item(),batch.max().item()))
    if trans:
        grids = np.transpose(grids.detach().cpu().numpy(), (1, 2, 0))
    plt.figure()
    plt.imshow(grids)
    if save:
        path='pic/'+name+'.png'
        plt.savefig(path)
    else:
       plt.show()

    # return grids












#device setting


device=torch.device("cuda:%d"%(gpu) if gpu_available else "cpu")


####################################################################################################



# data preprocessing

data_transform=transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()
    ])


dataset=datasets.ImageFolder(data_dir,data_transform)

dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2)

#get the 1st batch
# real_batch=next(iter(dataloader))

#
# #visualize the batch
# grids=vutils.make_grid(real_batch[0].to(device)[:], padding=0, normalize=True,nrow=4).cpu()
# transpose_matric=np.transpose(grids,(1,2,0))
#
# plt.figure()
# plt.imshow(transpose_matric)
# plt.show()



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        # nn.init.xavier_uniform(m.weight)
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



#init the net

#encoder (generator)
encoder=model.Encoder()
encoder=encoder.to(device)
encoder.apply(weights_init)

#decoder (discriminator)
decoder=model.Decoder()
decoder=decoder.to(device)
decoder.apply(weights_init)

#steganalyzer
steganalyzer = XuNet.Steganalyzer()
steganalyzer=steganalyzer.to(device)
steganalyzer.apply(weights_init)

# print(encoder)
# print(decoder)
# print(steganalyzer)


#define the loss

ssim_loss=pytorch_ssim.SSIM()
mssim_loss=pytorch_msssim.MSSSIM()
mse_loss=nn.MSELoss()
dis_loss=nn.BCELoss()


#define optimizer
optimizerD=optim.Adam(decoder.parameters(),lr=learning_rate,weight_decay=weight_decay)
optimizerG=optim.Adam(encoder.parameters(),lr=learning_rate,weight_decay=weight_decay)
optimizerS=optim.SGD(steganalyzer.parameters(),lr=learning_rate,weight_decay=weight_decay)


#list to variable the training
#


cover_ssmi=[]
secret_ssmi=[]
# cover_mssim=[]
# secret_mssim=[]
# covers_mse=[]
# secret_mse=[]
network_loss=[]
discrimin_loss=[]


##############################################################################################################

#loading checkpoint
if resume:
    print('loading params')
    start_epoch = params.start
    path = model_dir + '/' + str(start_epoch-1) + '.pth.tar'    #load the last epoch params
    checkpoint= torch.load(path,map_location='cpu' )
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['deocoder_state_dict'])
    steganalyzer.load_state_dict(checkpoint['stganalyzer_state_dict'])
    optimizerS.load_state_dict(checkpoint['stgan_optim'])
    optimizerD.load_state_dict(checkpoint['de_optim'])
    optimizerG.load_state_dict(checkpoint['en_optim'])
    epoch=checkpoint['epoch']
    cover_ssmi=checkpoint['cover_ssmi']
    secret_ssmi=checkpoint['secret_ssmi']
    network_loss=checkpoint['net_loss']
    encoder.train()
    decoder.train()
    steganalyzer.train()

    # assert start_epoch==start_epoch

else:
    start_epoch=1







##############################################################################################################

#train step








print('start training')

real=1
fake=0

for epoch in range(start_epoch,epochs):

#after 20 ,the lr is decreasing.
    if epoch>=20 and epoch %3==0:
        optimizerD.param_groups[0]['lr'] *= 0.9
        optimizerG.param_groups[0]['lr'] *= 0.9
        optimizerS.param_groups[0]['lr'] *= 0.9
        optimizerD.param_groups[0]['weight_decay'] *= 0.95
        optimizerG.param_groups[0]['weight_decay'] *= 0.95
        optimizerS.param_groups[0]['weight_decay'] *= 0.95



# cuz we need to figure out the MSE loss, so the target must be the same as the original one.
# That we should get two group different images as the covers and secret.
    for i,images in enumerate(dataloader):
        if len(images[0])!=8 : break
        covers=images[0][:4]
        secrets=images[0][4:]
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
        encoder.zero_grad()
        decoder.zero_grad()

        stegos=encoder(covers,secrets)

        reveals=decoder(stegos)

        #forward finish


##############################################################################################################
        _,f2_0,f4_0=steganalyzer(covers)


        cls_fake,f2_1,f4_1=steganalyzer(stegos)
        cls_fake=cls_fake.view(-1,2)
        cls_fake=cls_fake[:,0]
       # mean2 = cls_fake.mean().item()
        #####
        #####  discriminative loss of encoder
        real_label = torch.ones(cls_fake.size(), device=device)
        errG = dis_loss(cls_fake, real_label)


        f2_loss=mse_loss(f2_0,f2_1)
        f4_loss=mse_loss(f4_0,f4_1)
        perceptual_loss=(f2_loss+f4_loss)*lambda0
        #####
        #####similar loss of encoder


        # meanc=[covers.mean(),covers.min(),covers.max()]
        # means=[stegos.mean(),stegos.min(),stegos.max()]
        # print(meanc)
        # print(means)

        # mse1=mse_loss(covers,secrets0)
        # mse2=mse_loss(stegos,secrets0)
        # print(c_mse,mse1,mse2)

        s_mse = mse_loss(reveals, secrets)  ##### train the discriminator (steganalyzer)
        c_mse = mse_loss(stegos, covers)


        s_ssim=ssim_loss(reveals,secrets)
        c_ssim=ssim_loss(covers,stegos)

        if epoch==1 and i<=20:
            s_mssim=s_ssim
            c_mssim=c_ssim
        else:
            s_mssim=mssim_loss(secrets,reveals)
            c_mssim=mssim_loss(covers,stegos)
        # s_mssim = mssim_loss(secrets, reveals)
        # c_mssim = mssim_loss(covers, stegos)

        loss_c=alpha*(1-c_ssim)+(1-alpha)*(1-c_mssim)+beta*c_mse
        loss_s=alpha*(1-s_ssim)+(1-alpha)*(1-s_mssim)+beta*s_mse

        loss_similar=loss_c+gamma*loss_s
        loss=loss_similar+errG+perceptual_loss
        # print(loss_similar.cpu().item())
        # print(errG.cpu().item())

        loss.backward()
        optimizerG.step()
        optimizerD.step()


        # print("train g")

        ##### train the discriminator (steganalyzer)
        if i % 2 == 0:
            steganalyzer.zero_grad()
            stegos2 = stegos.detach()  # stegos2 is the stego without gradient to train the discriminator
            # output of the steganalyzer
            # output should be a softmax, so the shape is (batch,2)
            cls_real,_,__ = steganalyzer(covers)
            cls_fake,_,__ = steganalyzer(stegos2)
            cls_real=cls_real.view(-1,2)
            cls_fake=cls_fake.view(-1,2)
            # as calculating the loss, we just need the true label
            cls_real = cls_real[:, 0]
            cls_fake = cls_fake[:, 0]

            real_label = torch.ones(cls_real.size(), device=device)
            fake_label = torch.zeros(cls_fake.size(), device=device)

            # calculate the d_loss
            error_real = dis_loss(cls_real, real_label)
            error_fake = dis_loss(cls_fake, fake_label)

            # error_real.backward()
            # error_fake.backward()

            errors = error_fake + error_real
            errors.backward()
            optimizerS.step()



            # print("train d")

        ################################################################################
        ################################################################################
        #print the loss and something

        global_step+=1





        if i%1 ==0:
            print('epoch: %d || batch: %d || en_loss: %.4f ,  p_loss: %.4f || dis_loss: %.4f || ssim: %.4f|%.4f || mse: %.4f|%.4f || mssim:%.4f|%.4f '
                  % (epoch,i+1,loss,perceptual_loss,errors,s_ssim.item(),c_ssim.item(),s_mse.item(),c_mse.item(),s_mssim.item(),c_mssim.item()))

        if i%1==0:

            stego_result=torch.cat([covers,stegos],dim=0)

            reveal_result=torch.cat([secrets,reveals],dim=0)
            result_c = viz.images(stego_result*255, nrow=batch_size // 2, win= result_c,opts=dict(title="cover vs. stego"))
            result_s = viz.images(reveal_result, nrow=batch_size // 2, win=result_s,opts=dict(title="secret vs. reveal"))


            ###############################
            discrimin_loss.append(errors.item())
            cover_ssmi.append(c_ssim.item())
            secret_ssmi.append(s_ssim.item())
            network_loss.append(loss.item())

            a=plt.figure()
            plt.plot(discrimin_loss)
            plt.title("steganalyzer_loss")
            b=plt.figure()
            plt.plot(cover_ssmi)
            plt.title("cover & stego ssmi")
            c=plt.figure()
            plt.plot(secret_ssmi)
            plt.title("secret & reveal ssmi")
            d=plt.figure()
            plt.plot(network_loss)
            plt.title("network loss")

            lc_ssim = viz.matplot(a,win=lc_ssim)
            ls_ssim = viz.matplot(b,win=ls_ssim)
            l_dis = viz.matplot(c,win=l_dis)
            l_net = viz.matplot(d,win=l_net)






            if i%2==0:
                name = str(epoch) + '_' + str(i+1) + '_1'
                visualize_batch(stego_result, save=True, name=name)
                name = str(epoch) + '_' + str(i+1) + '_2'
                visualize_batch(reveal_result, save=True, name=name)









    if  epoch%2==0:
        path=model_dir+'/'+str(epoch)+'.pth.tar'
        torch.save({
            'epoch':epoch,
            'encoder_state_dict':encoder.state_dict(),
            'deocoder_state_dict':decoder.state_dict(),
            'stganalyzer_state_dict':steganalyzer.state_dict(),
            'en_optim':optimizerG.state_dict(),
            'de_optim':optimizerD.state_dict(),
            'stgan_optim':optimizerS.state_dict(),
            'cover_ssmi':cover_ssmi,
            'secret_ssmi':secret_ssmi,
            'net_loss':network_loss

        },path)


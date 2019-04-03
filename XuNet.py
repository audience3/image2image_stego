import torch
import torchvision
from torch import nn as nn

class Steganalyzer(nn.Module):
    def __init__(self):
        super(Steganalyzer,self).__init__()

        self.conv1=nn.Conv2d(3,8,3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(8)
        self.relu=   nn.LeakyReLU()
        self.pool=   nn.AvgPool2d(5,2)
        self.conv2=    nn.Conv2d(8,16,3,stride=1,padding=1)
        self.bn2=    nn.BatchNorm2d(16)
            # nn.LeakyReLU,
            # nn.AvgPool2d(5,2)


        self.block2=nn.Sequential(
            nn.Conv2d(16,32,1,stride=2,padding=0),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,1,stride=2,padding=0),
            nn.BatchNorm2d(64),

        )

        self.block3=nn.Sequential(
            nn.Conv2d(64,128,1,stride=2,padding=0),
            nn.BatchNorm2d(128)

        )


        kernel = [4, 2, 1]
        width = 8
        self.pool_4 = nn.MaxPool2d(kernel_size=width // kernel[0], stride=width // kernel[0])
        self.pool_2 = nn.MaxPool2d(kernel_size=width // kernel[1], stride=width // kernel[1])
        self.pool_1 = nn.MaxPool2d(kernel_size=width // kernel[2], stride=width // kernel[2])
        self.fc1 = nn.Linear(2688, 128)
        self.fc2 = nn.Linear(128, 2)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax()
        self.leaky=nn.LeakyReLU()


    def forward(self,stego):

        # conv op
        out1=self.conv1(stego)
        out=torch.abs(out1)  # the abs operation after the 1st conv in XuNet
        out=self.pool(self.relu(self.bn1(out)))
        out2=self.pool(self.relu(self.bn2(self.conv2(out))))
        out3=self.block2(out2)
        out4=self.block3(out3)
        out=self.leaky(out4)

        #spp pooling
        feature_4=self.pool_4(out)
        feature_2=self.pool_2(out)
        feature_1=self.pool_1(out)

        #reshape to 1dim
        feature_4=feature_4.view(len(stego),-1)
        feature_2 = feature_2.view(len(stego), -1)
        feature_1 = feature_1.view(len(stego), -1)

        spp_out=torch.cat([feature_4,feature_2,feature_1],dim=1)
        out=self.fc1(spp_out)
        out=self.fc2(out)
        out=self.softmax(out)




        return  out,out2,out4
from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
#==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#==============================================================================
#==============================================================================
class AttenResNet4(nn.Module):
    def __init__(self, atten_activation='sigmoid', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):

        super(AttenResNet4, self).__init__()

        self.pre = nn.Sequential( # channel expansion 
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
 
        if atten_channel == 1:  
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else: 
            self.conv1 = nn.Sequential( # channel compression 
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )
       
        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*20*3
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.apply(_weights_init)       
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        # print('skip4', skip4.size())# skip4=torch.Size([32, 16, 184, 243])
        x = self.att5(self.down5(x))
        # print(x.size())#torch.Size([32, 16, 182, 121])
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual 
        #print(x.size())
        
        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual 
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual 
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        # print("x.size() = ",x.size())
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        ##print(out.size()) 
        return out
        # return out, weight



#==============================================================================
class AttenResNet4_3(nn.Module):
    def se_gated_linearconcat_res2net50_v1b(self,**kwargs):
        model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
        return model
    def __init__(self, atten_activation='sigmoid', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):

        super(AttenResNet4, self).__init__()
        self.pre = nn.Sequential( # channel expansion 
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        #----------------------------------
        self.softmax = nn.Softmax(dim=3)
        #----------------------------------
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Softmax(dim=3)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Softmax(dim=3)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Softmax(dim=3)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Softmax(dim=3)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
 
        if atten_channel == 1:  
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else: 
            self.conv1 = nn.Sequential( # channel compression 
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )
       
        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*20*3
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()
        #=======================res2net===========================
        self.res2net = self.se_gated_linearconcat_res2net50_v1b(pretrained=False,loss='oc-softmax')
        #=======================res2net^===========================
   
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        # print("x.size() = ",x.size())
        skip3 = self.skip3(x)
        # print('skip3', skip3.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        # print('skip4', skip4.size())# skip4=torch.Size([32, 16, 184, 243])
        x = self.att5(self.down5(x))
        x = self.att6(skip4 * self.up5(x))
        # print("x = ",x.size())
        x = self.att7(skip3 * self.up4(x))
        x = self.att8(skip2 * self.up3(x))
        x = self.att9(skip1 * self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual # torch.Size([16, 1, 192, 251])
        out = self.res2net(x)
        return out
#===========================================================
#==============================================================================
class AttenResNet4_2(nn.Module):
    def se_gated_linearconcat_res2net50_v1b(self,**kwargs):
        model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
        return model
    def __init__(self, atten_activation='sigmoid', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):

        super(AttenResNet4, self).__init__()
        self.pre = nn.Sequential( # channel expansion 
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Sigmoid()
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Sigmoid()
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Sigmoid()
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.Sigmoid()
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
 
        if atten_channel == 1:  
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else: 
            self.conv1 = nn.Sequential( # channel compression 
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )
       
        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*20*3
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()
        #=======================res2net===========================
        self.res2net = self.se_gated_linearconcat_res2net50_v1b(pretrained=False,loss='oc-softmax')
        #=======================res2net^===========================
   
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        # print("x.size() = ",x.size())
        skip3 = self.skip3(x)
        # print('skip3', skip3.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        # print('skip4', skip4.size())# skip4=torch.Size([32, 16, 184, 243])
        x = self.att5(self.down5(x))
        # print(x.size())#torch.Size([32, 16, 182, 121])
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        # print("skip3.size() = ",skip3.size()," x.size() = ",x.size())
        # print("skip2.size() = ",skip2.size())
        # print("skip1.size() = ",skip1.size())

        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual # torch.Size([16, 1, 192, 251])
        out = self.res2net(x)
        return out
#===========================================================
class AttenResNet4_1(nn.Module):
    def se_gated_linearconcat_res2net50_v1b(self,**kwargs):
        model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
        return model
    def __init__(self, atten_activation='sigmoid', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):

        super(AttenResNet4, self).__init__()

        self.pre = nn.Sequential( # channel expansion 
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
 
        if atten_channel == 1:  
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else: 
            self.conv1 = nn.Sequential( # channel compression 
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )
       
        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*20*3
        
        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 2)
        self.sigmoid = nn.Sigmoid()
        #=======================res2net===========================
        self.res2net = self.se_gated_linearconcat_res2net50_v1b(pretrained=False,loss='oc-softmax')
        #=======================res2net^===========================
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        # print('skip4', skip4.size())# skip4=torch.Size([32, 16, 184, 243])
        x = self.att5(self.down5(x))
        # print(x.size())#torch.Size([32, 16, 182, 121])
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        # print("skip3.size() = ",skip3.size()," x.size() = ",x.size())
        # print("skip2.size() = ",skip2.size())
        # print("skip1.size() = ",skip1.size())

        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual # torch.Size([16, 1, 192, 251])
        out = self.res2net(x)
        return out
class AttenResNet4_cbam(nn.Module):
    def __init__(self, atten_activation='softmax3', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):
        super(AttenResNet4, self).__init__()
        #===============================================
        self.ca = ChannelAttention(atten_channel)
        self.sa = SpatialAttention()
        self.drop = nn.Dropout(0.5)
        #===============================================
        self.pre = nn.Sequential( # channel expansion 
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)
 
        if atten_channel == 1:  
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else: 
            self.conv1 = nn.Sequential( # channel compression 
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )
       
        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        
        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        
        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3)) # no padding 
        

        # # (N x 32 x 8 x 11) to (N x 32*8*11)
        # self.flat_feats = 32*20*3
        
        # # fc
        # self.ln1 = nn.Linear(self.flat_feats, 32)
        # self.bn7 = nn.BatchNorm1d(32)
        # self.re7 = nn.ReLU(inplace=True)
        # self.ln2 = nn.Linear(32, 32)
        # self.bn8 = nn.BatchNorm1d(32)
        # self.re8 = nn.ReLU(inplace=True)
        # self.ln3 = nn.Linear(32, 32)
        # ####
        # self.bn9 = nn.BatchNorm1d(32)
        # self.re9 = nn.ReLU(inplace=True)
        # self.ln4 = nn.Linear(32, 32)
        # self.bn10= nn.BatchNorm1d(32)
        # self.re10= nn.ReLU(inplace=True)
        # self.ln5 = nn.Linear(32, 32)
        # ###
        # self.bn11= nn.BatchNorm1d(32)
        # self.re11= nn.ReLU(inplace=True)
        # self.ln6 = nn.Linear(32, 2)
        # self.sigmoid = nn.Sigmoid()

   
    
    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
  

        x = self.att1(self.down1(self.pre(x)))# x.size() =  torch.Size([16, 16, 190, 249])
        skip1 = self.skip1(x)
        #print('size2', x.size())    
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        # print('skip4', skip4.size())# skip4=torch.Size([32, 16, 184, 243])
        x = self.att5(self.down5(x))
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.up1(x)
        # print("x1.size() = ",x.size())
        #cbam
        x = self.ca(x)*x
        x = self.sa(x)*x  
        x = self.conv1(x)
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual 
        #print(x.size())
        
        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual 
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual 
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual 
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual 
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual 
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        x = self.drop(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x = self.drop(x)
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x = self.drop(x)
        x += residual
        ###
        x = self.ln6(self.re11(self.bn11(x)))
        x = self.drop(x)
        out = self.sigmoid(x)
        ##print(out.size()) 
        return out
        # return out, weight
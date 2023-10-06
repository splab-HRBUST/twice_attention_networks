from __future__ import print_function
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
import math
from torch.nn import Parameter

#===================================res2Net====================================
class OCSoftmaxWithLoss(nn.Module):
    """
    OCSoftmaxWithLoss()
    
    """
    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = nn.Softplus()

    def forward(self, inputs, target):
        """ 
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from OCAngle
                 inputs[0]: positive class score
                 inputs[1]: negative class score
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # Assume target is binary, positive (genuine) = 0, negative (spoofed) = 1
        # 
        # Equivalent to select the scores using if-elese
        # if target = 1, use inputs[1]
        # else, use inputs[0]
        output = inputs[1] * target.view(-1, 1) + \
                 inputs[0] * (1-target.view(-1, 1))
        loss = self.m_loss(output).mean()
        return loss
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print('se reduction: ', reduction)
        # print(channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):   # x: B*C*D*T
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class LinearConcatGate(nn.Module):
    def __init__(self, indim, outdim):
        super(LinearConcatGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(indim, outdim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_prev, x):
        x_cat = torch.cat([x_prev, x], dim=1)
        b, c_double, _, _ = x_cat.size()
        c = int(c_double/2)
        y = self.avg_pool(x_cat).view(b, c_double)
        y = self.sigmoid(self.linear(y)).view(b, c, 1, 1)
        return x_prev * y.expand_as(x_prev)


class SEGatedLinearConcatBottle2neck(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 baseWidth=26,
                 scale=4,
                 stype='normal',
                 gate_reduction=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(SEGatedLinearConcatBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes,
                               width * scale,
                               kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(
                nn.Conv2d(width,
                          width,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False))
            bns.append(nn.BatchNorm2d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        if stype != 'stage':
            gates = []
            for i in range(self.nums-1):
                gates.append(LinearConcatGate(2*width, width))
            self.gates = nn.ModuleList(gates)

        self.conv3 = nn.Conv2d(width * scale,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SELayer(planes * self.expansion, reduction=16)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        # print('x: ', x.size())
        out = self.conv1(x)
        # print('conv1: ', out.size())
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = gate_sp + spx[i]
            sp = self.convs[i](sp)
            bn_sp = self.bns[i](sp)
            if self.stype != 'stage' and i < self.nums-1:
                gate_sp = self.gates[i](sp, spx[i+1])
            sp = self.relu(bn_sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # print('conv2: ', out.size())
        # print(self.width)
        # print(self.scale)
        out = self.conv3(out)
        # print('conv3: ', out.size())
        out = self.bn3(out)
        # print('bn3: ', out.size())
        out = self.se(out)
        # print('se :', out.size())

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class OCAngleLayer(nn.Module):
    """ Output layer to produce activation for one-class softmax

    Usage example:
     batchsize = 64
     input_dim = 10
     class_num = 2

     l_layer = OCAngleLayer(input_dim)
     l_loss = OCSoftmaxWithLoss()

     data = torch.rand(batchsize, input_dim, requires_grad=True)
     target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
     target = target.to(torch.long)

     scores = l_layer(data)
     loss = l_loss(scores, target)

     loss.backward()
    """
    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = w_posi
        self.w_nega = w_nega
        self.out_planes = 1
        
        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        #self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2,1,1e-5).mul_(1e5)

        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations
        
        input:
        ------
          input tensor (batchsize, input_dim)

        output:
        -------
          tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        inner_wx = input.mm(w)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        if flag_angle_only:
            pos_score = cos_theta
            neg_score = cos_theta
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
        out = torch.cat([neg_score,pos_score],dim=1)    
        return out
class GatedRes2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, m=0.35, num_classes=1000, loss='softmax', gate_reduction=4, **kwargs):
        self.inplanes = 16
        super(GatedRes2Net, self).__init__()
        self.loss = loss
        self.baseWidth = baseWidth
        self.scale = scale
        self.gate_reduction = gate_reduction
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, 1, 1, bias=False))
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])#64
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)#128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)#256
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)#512
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.stats_pooling = StatsPooling()

        if self.loss == 'softmax':
            # self.cls_layer = nn.Linear(2*8*128*block.expansion, num_classes)
            self.cls_layer = nn.Sequential(nn.Linear(128*block.expansion, num_classes), nn.LogSoftmax(dim=-1))
            self.loss_F = nn.NLLLoss()
        elif self.loss == 'oc-softmax':
            self.cls_layer = OCAngleLayer(128*block.expansion, w_posi=0.9, w_nega=0.2, alpha=20.0)
            self.loss_F = OCSoftmaxWithLoss()
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride,
                             stride=stride,
                             ceil_mode=True,
                             count_include_pad=False),
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample=downsample,
                  stype='stage',
                  baseWidth=self.baseWidth,
                  scale=self.scale,
                  gate_reduction=self.gate_reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      baseWidth=self.baseWidth,
                      scale=self.scale,
                      gate_reduction=self.gate_reduction))

        return nn.Sequential(*layers)

    def _forward(self, x):
        # x = x.unsqueeze(dim=1)
        # print("x1.size() = ",x.size())
        #x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        # print('maxpool: ', x.size())

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())
        # x = self.stats_pooling(x)
        x = self.avgpool(x)
        # print('avgpool:', x.size())
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        x = self.cls_layer(x)
        return x
        # return F.log_softmax(x, dim=-1)

    def extract(self, x):
        # x = x[:, None, ...]
        x = self.conv1(x)
        # print('conv1: ', x.size())
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print('layer1: ', x.size())
        x = self.layer2(x)
        # print('layer2: ', x.size())
        x = self.layer3(x)
        # print('layer3: ', x.size())
        x = self.layer4(x)
        # print('layer4: ', x.size())

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print('flatten: ', x.size())
        return x
    # Allow for accessing forward method in a inherited class
    forward = _forward



#=========================DY-ReluB=============================================
class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError
class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result
#==============================================================================
class AttenResNet4(nn.Module):
    def se_gated_linearconcat_res2net50_v1b(self,**kwargs):
        model = GatedRes2Net(SEGatedLinearConcatBottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
        return model
    def __init__(self, atten_activation='softmax3', atten_channel=16, size1=(192,251), size2=(190,249), size3=(188,247), size4=(186,245), size5=(184,243)):
    # def __init__(self, atten_activation='softmax3', atten_channel=16, size1=(513,401), size2=(511,399), size3=(509,397), size4=(507,395), size5=(505,393)):

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
            nn.BatchNorm2d(atten_channel),
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
            nn.BatchNorm2d(atten_channel),
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
            nn.BatchNorm2d(atten_channel),
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
            nn.BatchNorm2d(atten_channel),
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
                # nn.ReLU(inplace=True),
                DyReLUB(channels=atten_channel),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                # DyReLUB(channels=atten_channel//4),
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
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.bn_cqt = nn.BatchNorm2d(1)
        #=======================res2net===========================
        self.res2net = self.se_gated_linearconcat_res2net50_v1b(pretrained=False,loss='oc-softmax')
    
    def forward(self, x):
        """
        input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        x = x.unsqueeze(dim=1)
        # print("x.size() = ",x.size())
        ### feature
        #print('size1', x.size())
        residual = self.bn_cqt(x)
        # residual = x
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
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x1 = self.up1(x)
        # print("x1.size()=",x1.size())
        x = self.conv1(x1)
        # print("x1.size()=",x.size())
        weight = self.soft(x)
        #print(torch.sum(weight,dim=2)) # attention weight sum to 1
        # print("alpha = ",self.alpha)
        # print("weight = ",weight.device)
        x = (1 + self.alpha*weight) * residual # torch.Size([16, 1, 192, 251])
        out = self.res2net(x)
        return out
#===========================================================



























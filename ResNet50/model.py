
import torch
import torch.nn as nn
import sys

class Loss:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, cls, label):
        return self.loss(cls, label)
 
#-----------------------------------------------#
# 此处为定义3*3的卷积，即为指此次卷积的卷积核的大小为3*3
#-----------------------------------------------#
def conv3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
#-----------------------------------------------#
# 此处为定义1*1的卷积，即为指此次卷积的卷积核的大小为1*1
#-----------------------------------------------#
def conv1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
#----------------------------------#
# 此为resnet50中标准残差结构的定义
# conv3x3以及conv1x1均在该结构中被定义
#----------------------------------#
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #-----------------------------------------------#
        # 当步长的值不为1时,self.conv2 and self.downsample
        # 的作用均为对输入进行下采样操作
        # 下面为定义了一系列操作,包括卷积，数据归一化以及relu等
        #-----------------------------------------------#
        self.conv1      = conv1(inplanes, planes)
        self.bn1        = nn.BatchNorm1d(planes)
        self.conv2      = conv3(planes, planes, stride)
        self.bn2        = nn.BatchNorm1d(planes)
        self.conv3      = conv1(planes, planes * self.expansion)
        self.bn3        = nn.BatchNorm1d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
    #--------------------------------------#
    # 定义resnet50中的标准残差结构的前向传播函数
    #--------------------------------------#
    def forward(self, x):
        identity = x
        #-------------------------------------------------------------------------#
        # conv1*1->bn1->relu 先进行一次1*1的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv3*3->bn2->relu 先进行一次3*3的卷积之后进行数据归一化操作最后过relu增加非线性因素
        # conv1*1->bn3 先进行一次1*1的卷积之后进行数据归一化操作
        #-------------------------------------------------------------------------#
        out      = self.conv1(x)
        out      = self.bn1(out)
        out      = self.relu(out)
 
        out      = self.conv2(out)
        out      = self.bn2(out)
        out      = self.relu(out)
 
        out      = self.conv3(out)
        out      = self.bn3(out)
        #-----------------------------#
        # 若有下采样操作则进行一次下采样操作
        #-----------------------------#
        if self.downsample is not None:
            identity = self.downsample(identity)
        #---------------------------------------------#
        # 首先是将两部分进行add操作,最后过relu来增加非线性因素
        # concat（堆叠）可以看作是通道数的增加
        # add（相加）可以看作是特征图相加，通道数不变
        # add可以看作特殊的concat,并且其计算量相对较小
        #---------------------------------------------#
        out += identity
        out = self.relu(out)
 
        return out
 
#--------------------------------#
# 此为resnet50网络的定义
# input的大小为224*224
# 初始化函数中的block即为上面定义的
# 标准残差结构--Bottleneck
#--------------------------------#
class ResNet(nn.Module):
 
    def __init__(self, block, layers):
 
        super(ResNet, self).__init__()
        self.inplanes    = 64  # 初始信道数量设置为64
        self.block       = block  # 基础块
        #-----------------------------------#
        # conv1->bn1->relu
        # batch_size,1,205 -> batch_size,64,103
        #-----------------------------------#
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1   = nn.BatchNorm1d(self.inplanes)
        self.relu  = nn.ReLU(inplace=True)
        #------------------------------------#
        # maxpool
        # batch_size,64,103 -> batch_size,64,52
        #------------------------------------#
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        #------------------------------------#
        # Bottleneck
        # batch_size,64,52 -> batch_size,256,52
        #------------------------------------#
        self.layer1  = self._make_layer(block, 64, layers[0])
        #------------------------------------#
        # Bottleneck
        # batch_size,256,52 -> batch_size,512,26
        #------------------------------------#
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)
        #------------------------------------#
        # Bottleneck
        # batch_size,512,26 -> batch_size,1024,13
        #------------------------------------#
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)
        #------------------------------------#
        # Bottleneck
        # batch_size,1024,13 -> batch_size,2048,7
        #------------------------------------#
        self.layer4  = self._make_layer(block, 512, layers[3], stride=2)
        #--------------------------------------------#
        # AdaptiveAvgPool
        # batch_size,2048,7 -> batch_size,2048,1
        #--------------------------------------------#
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        #----------------------------------------#
        # Linear
        # batch_size,2048 -> batch_size,4
        #----------------------------------------#
        self.fc1     = nn.Linear(512 * block.expansion, 100)
        self.fc2     = nn.Linear(100, 4)
        self.loss_func = Loss()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample        = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        #--------------------------------------#
        # 按照x的第1个维度拼接（按照列来拼接，横向拼接）
        # 拼接之后,张量的shape为(batch_size,2048)
        #--------------------------------------#
        x = torch.flatten(x, 1)
        #--------------------------------------#
        # 过全连接层来调整特征通道数
        # (batch_size,2048)->(batch_size,4)
        #--------------------------------------#
        x = self.fc1(x)
        x = self.fc2(x)
        return x
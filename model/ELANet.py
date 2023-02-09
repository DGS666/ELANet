import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["ELANet"]


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):

        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):

        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BNPReLU(nn.Module):
    def __init__(self, nOut):

        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):

        output = self.bn(input)
        output = self.act(output)
        return output

class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):

        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):

        output = self.conv(input)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False)

    def forward(self, input):

        output = self.conv(input)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):

        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False, dilation=d)

    def forward(self, input):

        output = self.conv(input)
        return output

class ECG_D(nn.Module):
 

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
  
        super().__init__()
        self.conv1x1 = ConvBNPReLU(nIn, nOut, 3, 2)  
        self.conv1 = ConvBNPReLU(nOut, nOut, 1, 1)

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = Conv(2 * nOut, nOut, 1, 1)  

        self.CA = CCA(nOut, nOut)

    def forward(self, input):
        output1 = self.conv1x1(input)
        output = self.conv1(output1)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)

        output = joi_feat * self.CA(joi_feat) 

        return output


class ECG_R(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):

        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1) 
        self.conv1 = ConvBNPReLU(nIn + n, n, 1, 1)
        self.conv2 = ConvBNPReLU(nOut, nOut, 1, 1)
        self.F_loc1 = ChannelWiseConv(n, n, 3, 1) 
        self.F_sur1 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate)  
        self.F_loc2 = ChannelWiseConv(n, n, 3, 1)  
        self.F_sur2 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate) 
        self.bn_prelu1 = BNPReLU(nIn + n)
        self.bn_prelu2 = BNPReLU(nOut)
        self.add = add
        self.CA = CCA(nIn + n, nIn + n)

    def forward(self, input):
        output = self.conv1x1(input)
        loc1 = self.F_loc1(output)
        sur1 = self.F_sur1(output)

        x1 = loc1 + sur1
        x1 = torch.cat([input, x1], 1)
        x1 = self.bn_prelu1(x1)
        x1 = x1 * self.CA(x1)

        x2 = self.conv1(x1)
        loc2 = self.F_loc2(x2)
        sur2 = self.F_sur2(x2)

        x3 = torch.cat([loc2, sur2], 1)
        x4 = self.bn_prelu2(x3)
        output = self.conv2(x4)
        if self.add:
            output = input + output
        return output


class WDConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):

        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn,
                              bias=False, dilation=d)
        self.bnpre = BNPReLU(nIn)

    def forward(self, input):

        output = self.conv(input)
        return self.bnpre(output)


class CCA(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=(inchannel // 8 - 1), stride=inchannel // outchannel,
                      padding=(inchannel // 8 -2) // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(1, 1, kernel_size=(inchannel // 8 - 1), stride=1, padding=(inchannel // 8-2) // 2, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        out = self.pool(input)
        out = self.conv(out.squeeze(-1).transpose(-1, -2))
        out = self.sigmoid(out.transpose(-1, -2).unsqueeze(-1))
        return out


class SCA(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(SCA, self).__init__()
        self.conv = nn.Sequential(
            ConvBNPReLU(inchannel, inchannel//16, 1),
            ChannelWiseDilatedConv(inchannel//16, inchannel//16, 7),
            BNPReLU(inchannel//16),
            nn.Conv2d(inchannel//16, outchannel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return y



class RFF(nn.Module):
    def __init__(self, inchann, outchann, k_size):
        super().__init__()

        self.Xd1 = nn.Sequential(
            WDConv(inchann, inchann, k_size, 2, 1),
            nn.Conv2d(inchann, inchann * 2, 1, padding=0, bias=True),
            BNPReLU(inchann * 2),
        )
        self.Xd2_1 = nn.Sequential(
            WDConv(inchann * 2, inchann * 2, k_size, 1, 1),
            nn.Conv2d(inchann * 2, inchann * 2, 1, padding=0, bias=True),
            BNPReLU(inchann * 2),
        )
        self.Xd2 = WDConv(inchann * 2, inchann * 2, k_size, 1, 1)
        self.CA = CCA(128, 64)
        self.SA = SCA(128, 64)

        self.Xb_1 = nn.Sequential(
            nn.Conv2d(inchann * 8, inchann * 2, 1, padding=0, bias=True),
        )
        self.bnpre = BNPReLU(outchann)

    def forward(self, Xd1, Xd2, Xb):
        Xd1 = self.Xd1(Xd1)
        Xd2 = self.Xd2(Xd2)
        Xd2 = Xd1 + Xd2
        Xd2 = self.Xd2_1(Xd2)
        Xb = self.Xb_1(Xb)
        Xb = F.interpolate(Xb, Xd2.size()[2:], mode='bilinear',
                            align_corners=False)
        
        # DAF unit
        xcat = torch.cat([Xb, Xd2], 1)
        ca = self.CA(xcat)
        sa = self.SA(xcat)
        out1 = Xb * (sa + 1)
        out2 = Xd2 * (ca + 1)
        out = self.bnpre(torch.cat([out1, out2], 1))
        return out


class ELANet(nn.Module):


    def __init__(self, classes=19, M=2, N=5, dropout_flag=False):

        super().__init__()
        self.level1_0 = ConvBNPReLU(3, 32, 3, 2) 
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        self.b1 = BNPReLU(32)

        # stage 2
        self.level2_0 = ECG_D(32, 64, dilation_rate=2, reduction=8)
        self.level2 = nn.ModuleList()
        for i in range(0, M):
            self.level2.append(ECG_R(64, 64, dilation_rate=2, reduction=8))
        self.bn_prelu_2 = BNPReLU(128)

        # stage 3
        self.level3_0 = ECG_D(128, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.ModuleList()
        dilation_block = [4, 4, 4, 4, 4, 8, 8, 8, 8]
        for i in range(0, 2*N-1):
            self.level3.append(ECG_R(128, 128, dilation_rate=dilation_block[i], reduction=16))
        self.bn_prelu_3 = BNPReLU(256)

        self.decode = RFF(32, 128, 3)

        if dropout_flag:
            self.classifier = nn.Sequential(nn.Dropout2d(0.1, False), Conv(128, classes, 1, 1))
        else:
            self.classifier = nn.Sequential(Conv(128, classes, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find('ConvTranspose2d') != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):

        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)

        # stage 2
        output0_cat = self.b1(output0)
        output1_0 = self.level2_0(output0_cat) 

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
                
        # stage 4
        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))
        outputs = self.decode(output0_cat, output1, output2_cat)

        
        classifier = self.classifier(outputs)

      
        out = F.interpolate(classifier, input.size()[2:], mode='bilinear',
                            align_corners=False)  
        return out

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ELANet(classes=19).to(device)
    summary(model, (3, 360, 480))

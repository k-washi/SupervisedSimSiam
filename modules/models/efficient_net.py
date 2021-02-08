import torch
from torch import nn
from torch._C import dtype
import math

from modules.models.model_helper import Swish, weight_init

class SqueezeExcitation(nn.Module):
    """
    Squeeze Excitation module
    """

    def __init__(self, ch_in, ch_sq):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # channel x 1 x 1に圧縮
            nn.Conv2d(ch_in, ch_sq, 1), #in_channel, out_channel, kernel size, kearnel sizeが1なら、出力のsizeが変わらない => channel数が変わったのみ
            Swish(),
            nn.Conv2d(ch_sq, ch_in, 1),
        )

        self.se.apply(weight_init)
    
    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

class ConvBN(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size,
                        stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(ch_out),
        )
        self.layers.apply(weight_init)

    def forward(self, x):
        return self.layers(x)

class DropConnect(nn.Module):
    """
    BM modeuleの最終層で[stochastic depth](https://arxiv.org/pdf/1603.09382v2.pdf)をやるため
    """
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if self.training:
            keep_rate = 1.0 - self.drop_rate
            r = torch.rand([x.size(0), 1, 1, 1], dtype=x.dtype).to(x.device)
            r += keep_rate
            mask = r.floor()
            return x.div(keep_rate) * mask
        else:
            return x

class BMConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride, kernel_size, reduction_ratio=4, drop_connect_rate=0.2):
        super().__init__()
        self.use_residual = (ch_in == ch_out) & (stride == 1)
        ch_med = int(ch_in * expand_ratio)
        ch_sq = max(1, ch_in//reduction_ratio)

        # define net
        if expand_ratio != 1.0:
            layers = [ConvBN(ch_in, ch_med, 1), Swish()]
        else:
            layers = []
        
        layers.extend([
            ConvBN(ch_med, ch_med, kernel_size, stride=stride, padding=(kernel_size - 1)//2, groups=ch_med), ## depth-wise
            Swish(),
            SqueezeExcitation(ch_med, ch_sq),
            ConvBN(ch_med, ch_out, 1) # pixwise
        ])

        if self.use_residual:
            self.drop_connect = DropConnect(drop_connect_rate)
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.drop_connect(self.layers(x))
        
        else:
            return self.layers(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    

class EfficientNet(nn.Module):
    def __init__(self, width_mult=1., depth_mult=1., resolution=False, dropout_rate=0.2, input_ch=3, num_classes=1000):
        super().__init__()

        # expand_ratio, channel, repeats, stride, kernel_size                   
        settings = [
            [1,  16, 1, 1, 3],  # MBConv1_3x3, SE, 112 -> 112                   
            [6,  24, 2, 2, 3],  # MBConv6_3x3, SE, 112 ->  56                   
            [6,  40, 2, 2, 5],  # MBConv6_5x5, SE,  56 ->  28                   
            [6,  80, 3, 2, 3],  # MBConv6_3x3, SE,  28 ->  14                   
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE,  14 ->  14                   
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE,  14 ->   7                   
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE,   7 ->   7]                  
        ]

        ch_out = int(math.ceil(32 * width_mult))
        features = [nn.AdaptiveAvgPool2d(resolution)] if resolution else []
        features.extend([ConvBN(input_ch, ch_out, 3, stride=2), Swish()])

        ch_in = ch_out
        for t, c, n, s, k in settings:
            ch_out = int(math.ceil(c*width_mult))
            repeats = int(math.ceil(n*depth_mult))

            for i in range(repeats):
                stride = s if i == 0 else 1
                features.extend([BMConvBlock(ch_in, ch_out, t, stride, k)])
                ch_in = ch_out
            
        ch_last = int(math.ceil(1280 * width_mult))
        features.extend([ConvBN(ch_in, ch_last, 1), Swish()])

        self.features = nn.Sequential(*features)

        middle_liner_output = ch_last // 2 
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.Linear(ch_last, middle_liner_output), # this is from supervised contrastive learning paper, not Effnet
                nn.Dropout(dropout_rate),
                nn.Linear(middle_liner_output, num_classes)
            )
        else:
            # this is feature vector
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def _efficientnet(w_mult, d_mult, resolution, drop_rate, input_ch, num_classes=1000):
    model = EfficientNet(w_mult, d_mult, resolution, drop_rate, input_ch, num_classes)
    return model


def efficientnet_b0(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.0, 224, 0.2)              
    return _efficientnet(1.0, 1.0, None, 0.2, input_ch, num_classes)

def efficientnet_b1(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.0, 1.1, 240, 0.2)              
    return _efficientnet(1.0, 1.1, None, 0.2, input_ch, num_classes)

def efficientnet_b2(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.1, 1.2, 260, 0.3)              
    return _efficientnet(1.1, 1.2, None, 0.3, input_ch, num_classes)

def efficientnet_b3(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.2, 1.4, 300, 0.3)              
    return _efficientnet(1.2, 1.4, None, 0.3, input_ch, num_classes)

def efficientnet_b4(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.4, 1.8, 380, 0.4)              
    return _efficientnet(1.4, 1.8, None, 0.4, input_ch, num_classes)

def efficientnet_b5(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.6, 2.2, 456, 0.4)              
    return _efficientnet(1.6, 2.2, None, 0.4, input_ch, num_classes)

def efficientnet_b6(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (1.8, 2.6, 528, 0.5)              
    return _efficientnet(1.8, 2.6, None, 0.5, input_ch, num_classes)

def efficientnet_b7(input_ch=3, num_classes=1000):
    #(w_mult, d_mult, resolution, droprate) = (2.0, 3.1, 600, 0.5)              
    return _efficientnet(2.0, 3.1, None, 0.5, input_ch, num_classes)


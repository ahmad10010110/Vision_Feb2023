import time
import torch
import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode

class CBS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBS, self).__init__()
        
        conv = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        self.cbs = nn.Sequential(
            conv, 
            bn, 
            nn.SiLU(inplace=True)            
            )
        
    def forward(self, x):
        return self.cbs(x)
    
    
    
class Bottleneck(nn.Module):
    def __init__ (self, in_channels, out_channels, width_multiple=1):
        c_ = int(width_multiple*in_channels)
        
        self.c1 = CBS(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBS(c_, out_channels, kernel_size=3, stride=1, padding = 1)
        
    def forward(self, x):
        return self.c2(self.c1)
    
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        
        c_ = int(width_multiple * in_channels)
        
        self.c1 = CBS(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBS(in_channels, c_, kernel_size=1, stride=1, padding=0)
        
        if backbone:
            self.seq = nn.Sequential(
                
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
                )
            
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBS(c_, c_, 1, 1, 0),
                    CBS(c_, c_, 3, 1, 0)
                    ) for _ in range(depth)]
                )
        self.c_out = CBS(c_*2, out_channels, kernel_size=1, stride=1, padding=0)
        
        def forward(self, x):
            x = torch.cat([self.seq(self.c1(x)), self.c_skipped(self.c1(x))], dim=1)
            return self.c_out(x)
        
        

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        
        c_ = int(in_channels // 2)
        
        self.c1 = CBS(in_channels, c_, 1, 1, 2)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding = 2)
        self.c_out = CBS(c_ * 4, out_channels, 1, 1, 0)
        
    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        
        return self.c_out(torch.cat([x, pool1, pool2, pool3]) ,dim=1)


class HEADS(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        self.nc = nc
        self.nl = len(anchors)
        self.naxs = len(anchors[0])
        
        self.stride =[8, 16, 32]
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride)
        for in_channels in ch:
            self.out_convs +=[
                nn.Conv2d(in_channels=in_channels, out_channels=(5 + self.nc)*self.naxs, kernel_size=1)
                ]
    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x)
            
            
            
            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x)
        return x
    
    
class YOLOV5M(nn.Module):
    def __init__(self, first_out, nc=80, anchors=(), ch=()):
        self.backbone +=[
            
            CBS(in_channels=3, out_channels=first_out, kernel_size=6, stride=2, padding=2),
            CBS()
            C3(in_channels=first_out*2, out_channels=first_out*2, width_multiple=0.5, depth=2)
            CBS()
            C3()
            CBS()
            C3()
            CBS()
            C3()
            SPPF() ]
        
        self.neck +=[
            CBS()
            C3()
            CBS()
            C3()
            ]
            
    def forward(nn.Module):
        backbone_connection =[]
        neck_connection =[]
        
        output = []
        
        for idx, layer in enumerate(self.backbone):
            if idx in [4, 6]:
                x = layer(x)
                backbone_connection.append(x)
            
        for idx, layer in enumerate(self.neck):
            if idx in [0, 2, 3]:
                x = layer(x)
                neck_connection.append(x)
                
                
        
        outputs = [torch.cat([]), torch.cat([]), torch.cat[])
                   
        output = HEADS()
        
        
        
        
        
        
        return output
                
            
            
            
            ]
            
        
        
        
        
        
    
        
    
    
    
    
    
    
    
    
    
    
    
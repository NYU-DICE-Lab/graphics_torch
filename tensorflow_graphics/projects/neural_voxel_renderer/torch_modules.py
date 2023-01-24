import torch
import torch.nn as nn

from torch_helpers import load_params
from torch_blocks import ConvBlock3d, ResBlock3d

class VoxelProcessing(nn.Module):
    def __init__(self):
        super(VoxelProcessing,self).__init__()
        nf_2d = 512
        self.vol0_a = ConvBlock3d(4, 16, size=4, strides=2)
        self.vol0_b = ConvBlock3d(16, 16, size=4, strides=1)
        self.vol1_a = ConvBlock3d(16, 16, size=4, strides=2)
        self.vol1_b = ConvBlock3d(16, 32, size=4, strides=1)
        self.vol1_c = ConvBlock3d(32, 32, size=4, strides=1)
        self.conv3ds = [self.vol0_a, self.vol0_b, self.vol1_a, self.vol1_b, self.vol1_c]
        
        self.vol_a1 = ResBlock3d(32, 32)
        self.vol_a2 = ResBlock3d(32, 32)
        self.vol_a3 = ResBlock3d(32, 32)
        self.vol_a4 = ResBlock3d(32, 32)
        self.vol_a5 = ResBlock3d(32, 32)
        
        self.resblocks = [self.vol_a1, self.vol_a2, self.vol_a3, self.vol_a4, self.vol_a5]
        
        self.vol_encoder = nn.Conv2d(32, nf_2d, 1, 1, 0)
        self.final_relu = nn.LeakyReLU(negative_slope=0.3)
        
        self.parameterize()
        
    def forward(self, voxels):
        voxels = self.vol0_a(voxels)
        voxels = self.vol0_b(voxels)
        voxels = self.vol1_a(voxels) 
        voxels = self.vol1_b(voxels) 
        voxels = self.vol1_c(voxels) # 0.9999977
        
        shortcut = voxels
        
        voxels = self.vol_a1(voxels) # 0.9905581
        voxels = self.vol_a2(voxels) # 0.9871442
        voxels = self.vol_a3(voxels)
        voxels = self.vol_a4(voxels)
        voxels = self.vol_a5(voxels)
        
        voxels = voxels + shortcut
        voxels = voxels.permute(0,2,3,4,1)
        voxels = voxels.reshape(voxels.shape[0],32,32,-1)
        voxels = voxels.permute(0,3,1,2)
        voxels = self.vol_encoder(voxels)
        voxels = self.final_relu(voxels)
        return voxels
    
    def parameterize(self):
        params = load_params()
        params_dict = {k:v for k,v in params}
        i=0
        for conv in self.conv3ds:
            if i==0:
                conv.parameterize(params_dict,'')
            else:
                conv.parameterize(params_dict,'_%s' % i)
            i+=1
            
        for res in self.resblocks:
            res.parameterize(params_dict,i)
            i+=2
            
        self.vol_encoder.weight = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/VoxelProcessing/conv2d/kernel']).permute(3,2,0,1))
        self.vol_encoder.bias = torch.nn.parameter.Parameter(torch.from_numpy(params_dict['Network/VoxelProcessing/conv2d/bias']))

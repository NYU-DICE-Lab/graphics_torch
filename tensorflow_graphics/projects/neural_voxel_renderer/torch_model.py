import torch
import torch.nn as nn
import numpy as np
import math

class ResBlock3d(nn.Module):
    
    def __init__(self,input_dim,nfilters):
        super(ResBlock3d,self).__init__()
        self.conv_1 = nn.Conv3d(input_dim,nfilters,3,stride=1,padding=1,bias=True)
        self.bn1 = nn.BatchNorm3d(nfilters)
        
        self.conv_2 = nn.Conv3d(nfilters,nfilters,3,stride=1,padding=1,bias=True)
        self.bn2 = nn.BatchNorm3d(nfilters)
        
        self.relu = nn.LeakyReLU(negative_slope=0.3)

        self.convs = [self.conv_1,self.conv_2]
        self.bns = [self.bn1,self.bn2]
    
    def forward(self,x):
        shortcut = x
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv_2(x) # 0.9887371
        x = self.bn2(x) # 0.9894749
        x = x + shortcut # 0.99114084
        
        x = self.relu(x)
        return x
        
    
    def parameterize(self,params_dict,layer_id):
        module = 'Network/VoxelProcessing'
        for i,conv in enumerate(self.convs):
            conv_weight = params_dict[module+'/conv3d_%s/kernel' % (layer_id+i)]
            conv_weight = torch.from_numpy(conv_weight).permute(4,3,0,1,2)
            conv_weight = torch.nn.parameter.Parameter(conv_weight)
            
            assert conv_weight.shape == conv.weight.shape
            conv.weight = conv_weight

            conv_bias = params_dict[module+'/conv3d_%s/bias' % (layer_id+i)]
            conv_bias = torch.from_numpy(conv_bias)
            conv_bias = torch.nn.parameter.Parameter(conv_bias)
            
            assert conv_bias.shape == conv.bias.shape
            conv.bias = conv_bias
            
        for i,bn in enumerate(self.bns):
            bn.running_mean = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_mean' % (layer_id+i)])
            bn.running_var = torch.from_numpy(params_dict[module+'/batch_normalization_%s/moving_variance' % (layer_id+i)])
            
            tf_gamma = torch.from_numpy(params_dict['%s/batch_normalization_%s/gamma' % (module,layer_id+i)])
            tf_beta = torch.from_numpy(params_dict['%s/batch_normalization_%s/beta' % (module,layer_id+i)])
            
            tf_gamma = torch.nn.parameter.Parameter(tf_gamma)
            tf_beta = torch.nn.parameter.Parameter(tf_beta)
            
            bn.weight = tf_gamma
            bn.bias = tf_beta

class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, nfilters, size, strides,
                  alpha_lrelu=0.2, normalization='None', relu=True):
        super(ConvBlock3d, self).__init__()

        # set padding so that output shape is the same as input shape
        padding = (size-1)//2
        self.stride = strides
        
        self.conv = nn.Conv3d(in_channels, nfilters, size, strides, padding=padding, padding_mode="zeros",bias=False)
        self.bn = nn.BatchNorm3d(nfilters)
        self.relu = nn.LeakyReLU(negative_slope=alpha_lrelu)

    def forward(self, x):
        if self.stride == 1:
            x=nn.functional.pad(x,(0,1,0,1,0,1),"constant",0)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
    def parameterize(self,params,layer_id):
        # setting convolution weights
        module = 'Network/VoxelProcessing'
        conv_weight = params[module+'/conv3d%s/kernel' % layer_id]
        conv_weight = torch.from_numpy(conv_weight).permute(4,3,0,1,2)
        conv_weight = torch.nn.parameter.Parameter(conv_weight)
        
        assert conv_weight.shape == self.conv.weight.shape
        self.conv.weight = conv_weight
        
        # setting batch norm params
        self.bn.running_mean = torch.from_numpy(params[module+'/batch_normalization%s/moving_mean' % layer_id])
        self.bn.running_var = torch.from_numpy(params[module+'/batch_normalization%s/moving_variance' % layer_id])
        
        tf_gamma = torch.from_numpy(params['%s/batch_normalization%s/gamma' % (module,layer_id)])
        tf_beta = torch.from_numpy(params['%s/batch_normalization%s/beta' % (module,layer_id)])
        
        tf_gamma = torch.nn.parameter.Parameter(tf_gamma)
        tf_beta = torch.nn.parameter.Parameter(tf_beta)
        
        self.bn.weight = tf_gamma
        self.bn.bias = tf_beta

class VoxelProcessing(nn.Module):
    def __init__(self):
        super().__init__()
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
    
def run_forward():
    model = VoxelProcessing()
    model.parameterize()
    params = load_params()
    
    
    # #load .npy files from test_data
    final_composite = np.load('test_data/final_composite.npy')
    interpolated_voxels = np.load('test_data/interpolated_voxels.npy')
    
    
    
    final_composite = torch.from_numpy(final_composite)
    interpolated_voxels = torch.from_numpy(interpolated_voxels).permute(0,4,1,2,3)
    
    model.eval()
    debug_output = model(interpolated_voxels)
    to_txt(debug_output)
    print('break')
    
def to_txt_3d(out):
    permuted = out.permute(0,2,3,4,1).detach().cpu().numpy()
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, permuted)
        
    # #all possible length 3 combinations of 10 and 20 
    indices = [[10,10,10],[10,10,20],[10,20,10],[10,20,20],[20,10,10],[20,10,20],[20,20,10],[20,20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(permuted))
        f.write('Standard deviation: %s \n' % np.std(permuted))
        for i,j,k in indices:
            f.write('index: %s,%s,%s' % (i,j,k))
            f.write('\n')
            f.write(str(permuted[0,i,j,k,:8]))
            f.write('\n')

def to_txt(out):
    if len(out.shape)==5:
        to_txt_3d(out)
        return
    permuted = out.permute(0,2,3,1).detach().cpu().numpy()
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, permuted)
        
    # #all possible length 3 combinations of 20 and 40
    # indices = [[20,20,20],[20,20,40],[20,40,20],[20,40,40],[40,20,20],[40,20,40],[40,40,20],[40,40,40]]
    #all possible length 2 combinations of 10 and 20 
    indices = [[10,10],[10,20],[20,10],[20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(permuted))
        f.write('Standard deviation: %s \n' % np.std(permuted))
        for i,j in indices:
            f.write('index: %s,%s' % (i,j))
            f.write('\n')
            f.write(str(permuted[0,i,j,:8]))
            f.write('\n')
        
def load_params():
    #load param_vals.npy from current directory
    file_path = 'param_vals.npy'
    with open(file_path, 'rb') as f:
        param_vals = np.load(f,allow_pickle=True)

    param_vals = list(param_vals)
    del param_vals[848]
    return param_vals

if __name__=="__main__":
    run_forward()
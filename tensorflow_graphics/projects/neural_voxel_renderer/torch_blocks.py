import torch
import torch.nn as nn

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

import torch
import torch.nn as nn
import numpy as np
import math

from torch_helpers import load_params,to_txt
from torch_modules import VoxelProcessing,ProjectionProcessing


class NVR_Plus(nn.Module):
    
    def __init__(self):
        super(NVR_Plus, self).__init__()
        self.voxel_processing = VoxelProcessing()
        self.projection_processing = ProjectionProcessing()
        
    def forward(self,voxels,final_composite):
        voxel_representation = self.voxel_processing(voxels)
        projection_representation = self.projection_processing(voxel_representation)
        
        
        return projection_representation
    
def run_forward():
    model = NVR_Plus()
    params = load_params()
    
    d=model.state_dict()
    
        
    # #load .npy files from test_data
    final_composite = np.load('test_data/final_composite.npy')
    interpolated_voxels = np.load('test_data/interpolated_voxels.npy')
    
    
    
    final_composite = torch.from_numpy(final_composite)
    interpolated_voxels = torch.from_numpy(interpolated_voxels).permute(0,4,1,2,3)
    
    model.eval()
    debug_output = model(interpolated_voxels,final_composite)
    to_txt(debug_output)
    print('break')
    
        

if __name__=="__main__":
    run_forward()
import numpy as np

# load torch output
with open('debug_outputs/torch_debug.npy', 'rb') as f:
    torch_output = np.load(f)
    
#load tf output
with open('debug_outputs/tf_debug.npy', 'rb') as f:
    tf_output = np.load(f)
    
# compute cossine similarity of the two outputs
cos_sim = np.dot(torch_output.flatten(),tf_output.flatten()) / (np.linalg.norm(torch_output.flatten()) * np.linalg.norm(tf_output.flatten()))
print('COSINE SIMILARITY:',cos_sim)
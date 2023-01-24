import tensorflow as tf
import numpy as np

from tensorflow_graphics.projects.neural_voxel_renderer import helpers
from tensorflow_graphics.projects.neural_voxel_renderer import models
from tensorflow_graphics.rendering.volumetric import visual_hull

import matplotlib.pyplot as plt

light_position = np.array([-1.0901234 ,  0.01720496,  2.6110773 ]).astype(np.float32)
light_position = np.expand_dims(light_position,axis=(0)).astype(np.float32)

VOXEL_SIZE = 128
IMAGE_SIZE = 256
GROUND_COLOR = np.array((136., 162, 199))/255.
BLENDER_SCALE = 2
DIAMETER = 4.2  # The voxel area in world coordinates

device = 'cuda:1'

latest_checkpoint = '/tmp/checkpoint/model.ckpt-126650'
tf.compat.v1.reset_default_graph()
g = tf.compat.v1.Graph()

with tf.device('/device:GPU:0'):
    with g.as_default():
        vol_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 4],
                                            name='input_voxels')
        rerender_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                                name='rerender')
        light_placeholder = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, 3],
                                            name='input_light')
        
        upstream_grad = tf.compat.v1.placeholder(tf.float32,
                                            shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3],
                                            name='upstream_grad')
        
        model = models.neural_voxel_renderer_plus(vol_placeholder,
                                                    rerender_placeholder,
                                                    light_placeholder)
        predicted_image_logits,debug_output = model.outputs
        
        adjusted = tf.reduce_sum(predicted_image_logits*upstream_grad)
        vol_gradients = tf.gradients(adjusted,vol_placeholder)
        rerender_gradients = tf.gradients(adjusted,rerender_placeholder)
        
        saver = tf.compat.v1.train.Saver()


def render_tf_forward(final_composite,interpolated_voxels):
    # tf.compat.v1.reset_default_graph()
    batch_size = interpolated_voxels.shape[0]
    a = interpolated_voxels
    b = final_composite*2.-1 #TODO account for this
    c = np.stack(batch_size*[light_position.squeeze()])
    d = np.zeros((batch_size,256,256,3))
    with tf.compat.v1.Session(graph=g,config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:
        devices = sess.list_devices()
        saver.restore(sess, latest_checkpoint)
        feed_dict = {vol_placeholder: a,
                    rerender_placeholder: b,
                    light_placeholder: c,
                    upstream_grad: d}
        out = sess.run(debug_output, feed_dict)
        to_text(out)
        print('break')
        predictions = sess.run(predicted_image_logits, feed_dict)

    return predictions

def to_text_3d(out):
    with open("debug_outputs/torch_debug.npy", "wb") as f:
        np.save(f, out)
        
    # #all possible length 3 combinations of 10 and 20 
    indices = [[10,10,10],[10,10,20],[10,20,10],[10,20,20],[20,10,10],[20,10,20],[20,20,10],[20,20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(out))
        f.write('Standard deviation: %s \n' % np.std(out))
        for i,j,k in indices:
            f.write('index: %s,%s,%s' % (i,j,k))
            f.write('\n')
            f.write(str(out[0,i,j,k,:8]))
            f.write('\n')

def to_text(out):
    if len(out.shape) == 5:
        to_text_3d(out)
        return
    with open("debug_outputs/tf_debug.npy", "wb") as f:
        np.save(f, out)
    
    indices = [[10,10],[10,20],[20,10],[20,20]]
    with open("torch_debug.txt", "w") as f:
        f.write('Mean: %s \n' % np.mean(out))
        f.write('Standard deviation: %s \n' % np.std(out))
        for i,j in indices:
            f.write('index: %s,%s' % (i,j))
            f.write('\n')
            f.write(str(out[0,i,j,:8]))
            f.write('\n')

def test():
    #load data from test_data in .npy format
    final_composite = np.load('test_data/final_composite.npy')
    final_composite = (final_composite - 0.5)*2
    interpolated_voxels = np.load('test_data/interpolated_voxels.npy')

    predictions = render_tf_forward(final_composite,interpolated_voxels)
    #save prediction as image using matplotlib
    preds = np.clip(predictions[0]*0.5+0.5,0,1)
    plt.imsave('test_data/prediction.png',preds)

if __name__ == '__main__':
    test()
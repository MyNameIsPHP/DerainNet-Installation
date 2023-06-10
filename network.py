
import tensorflow as tf
from GuidedFilter import guided_filter



num_feature = 512            # number of feature maps
num_channels = 3             # number of input's channels 
patch_size = 64              # patch size 

# DerainNet
def inference(images):
    with tf.variable_scope('DerainNet', reuse=tf.AUTO_REUSE): 
      
        base = guided_filter(images,images, 15, 1, nhwc=True) # using guided filter for obtaining base layer
        detail = images - base   # detail layer
        
        conv1  = tf.layers.conv2d(detail, num_feature, 16, padding="valid", activation = tf.nn.relu)
        conv2  = tf.layers.conv2d(conv1, num_feature, 1, padding="valid", activation = tf.nn.relu)
        output = tf.layers.conv2d_transpose(conv2, num_channels, 8, strides = 1, padding="valid")

    return output, base
  

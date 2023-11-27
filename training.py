# -*- coding: utf-8 -*-
#!/usr/bin/env python2



# This is a re-implementation of training code of this paper:
# X. Fu, J. Huang, X. Ding, Y. Liao and J. Paisley. “Clearing the Skies: A deep network architecture for single-image rain removal”, 
# IEEE Transactions on Image Processing, vol. 26, no. 6, pp. 2944-2956, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from network import inference
from GuidedFilter import guided_filter
import argparse


parser = argparse.ArgumentParser(description="ddn_train")
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--iterations", type=int, default=120000, help="Number of training iteration")
parser.add_argument("--save_path", type=str, default="./model/Rain100L/", help='path to save models and log files')
parser.add_argument("--data_path",type=str, default="./TrainData/Rain100L",help='path to training data')

opt = parser.parse_args()

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

##################### Network parameters ###################################
patch_size = 64              # patch size 
learning_rate = 1e-3         # learning rate
iterations = opt.iterations   # iterations
batch_size = opt.batch_size    # batch size
save_model_path = opt.save_path # saved model's path
model_name = 'model-epoch'   # saved model's name
############################################################################

input_path = opt.data_path + "/input/"    # the path of rainy images
gt_path = opt.data_path + "/label/"       # the path of ground truth



# randomly select image patches
def _parse_function(filename, label):  
     
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0
  
  
  image_string = tf.read_file(label)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  label = tf.cast(image_decoded, tf.float32)/255.0

  t = time.time()
  rainy = tf.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
  label = tf.random_crop(label, [patch_size, patch_size ,3],seed = t)   
  return rainy, label 


if __name__ == '__main__':

   RainName = os.listdir(input_path)
   RainName.sort(key=lambda x: int(re.search(r'\d+', x.split('_')[0]).group()))
   print(RainName)  
   for i in range(len(RainName)):
      RainName[i] = input_path + RainName[i]
   LabelName = os.listdir(gt_path)
   LabelName.sort(key=lambda x: int(re.search(r'\d+', x).group()))  # Sort based on the whole numeric part
   
   for i in range(len(LabelName)):
       LabelName[i] = gt_path + LabelName[i] 
   
   LabelName_ = []
   
   if (len(RainName) != len(LabelName)):
      
      multi = int(len(RainName) / len(LabelName))
      for i in range(len(LabelName)):
         for j in range(multi):
            LabelName_.append(LabelName[i])
   else:
      LabelName_ = LabelName
   
   filename_tensor = tf.convert_to_tensor(RainName, dtype=tf.string)  
   labels_tensor = tf.convert_to_tensor(LabelName_, dtype=tf.string) 
   
   # if filename_tensor != labels_tensor
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=batch_size * 10)
   dataset = dataset.batch(batch_size).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   rainy, labels = iterator.get_next()  
     
   details_label = labels - guided_filter(labels, labels, 15, 1, nhwc=True)
   details_label = details_label[:, 4:patch_size-4, 4:patch_size-4, :] # output size 56
  
   details_output, _ = inference(rainy)
  
   loss = tf.reduce_mean(tf.square( details_label - details_output ))  # MSE loss
   

   all_vars = tf.trainable_variables() 
   g_optim = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list = all_vars) # optimizer
   print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))  
   saver = tf.train.Saver(var_list = all_vars, max_to_keep = 5)

   config = tf.ConfigProto()
   config.gpu_options.allow_growth = True
  

   init =  tf.group(tf.global_variables_initializer(), 
                    tf.local_variables_initializer())
   
   with tf.Session(config=config) as sess:
       
      with tf.device('/gpu:0'): 
        sess.run(init)	
        tf.get_default_graph().finalize()     
                
        if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
               ckpt = tf.train.latest_checkpoint(save_model_path)
               saver.restore(sess, ckpt)  
               ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
               start_point = int(ckpt_num[-1]) + 1 
               print("Load success")
       
        else:  # re-training when no models found
               start_point = 0   
               print("re-training")


        check_data, check_label = sess.run([rainy, labels])
        print("Check patch pair:")  
        plt.subplot(1,2,1)     
        plt.imshow(check_data[0,:,:,:])
        plt.title('input')         
        plt.subplot(1,2,2)    
        plt.imshow(check_label[0,:,:,:])
        plt.title('ground truth')        
        plt.show()

        start = time.time()  

        for j in range(start_point, iterations):   # iterations
            
            _,Training_Loss = sess.run([g_optim,loss]) # training
      
            if np.mod(j+1,100) == 0 and j != 0: # save the model every 100 iterations    
               end = time.time()
               print ('%d / %d iteraions, Training Loss  = %.4f, runtime = %.1f s' % (j+1, iterations, Training_Loss, (end - start)))         
               save_path_full = os.path.join(save_model_path, model_name)
               saver.save(sess, save_path_full, global_step = j+1, write_meta_graph=False)
               start = time.time()      
               
        print('Training is finished.')
   sess.close()  
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow import keras
import tensorflow as tf
import os
import cv2
from tensorflow.keras import layers
from hyper_parameters import *




def Build_Model():
    
  input_layer = tf.keras.Input(shape=(IMAGE_H, IMAGE_W, 3), name='original_img')
  x = layers.Conv2D(16, 5,padding='same', activation='linear',name='conv1')(input_layer)
  x = layers.BatchNormalization(name='batch_norm1')(x)
  x = layers.LeakyReLU(alpha=0.1)(x)
  x = layers.MaxPooling2D(2)(x)
  
  x = layers.Conv2D(32, 3,padding='same', activation='linear',name='conv2')(x)
  x = layers.BatchNormalization(name='batch_norm2')(x)
  x = layers.LeakyReLU(alpha=0.1)(x)
  x = layers.MaxPooling2D(2)(x)

  x = layers.Conv2D(64, 3,padding='same', activation='linear',name='conv3')(x)
  x = layers.BatchNormalization(name='batch_norm3')(x)
  x = layers.LeakyReLU(alpha=0.1)(x)
  x = layers.MaxPooling2D(2)(x)

  x = layers.Conv2D(128, 3,padding='same', activation='linear',name='conv4')(x)
  x = layers.BatchNormalization(name='batch_norm4')(x)
  x = layers.LeakyReLU(alpha=0.1)(x)
  x = layers.MaxPooling2D(2)(x)

  x = layers.Conv2D(256, 3,padding='same', activation='linear',name='conv5')(x)
  x = layers.MaxPooling2D(2)(x)

  x = layers.Conv2D(NUM_ANCHOR*(5+NUM_CLASSES), 1,padding='same', activation='linear',name='conv6')(x)
  output = layers.Conv2D(NUM_ANCHOR*(5+NUM_CLASSES), 1,padding='same', activation='linear',name='conv7')(x)
  
#  output = layers.Dense(GRID_DIM*GRID_DIM*NUM_ANCHOR*(5+NUM_CLASSES))(x)
  model =  tf.keras.Model(input_layer,output, name='yolo_model')

  return model

Molo = Build_Model()
print (Molo.summary())
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

checkpoint_dir = './'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt_molo")

def compute_loss(labels, pred):
    
    loss_0 = compute_loss_no_obj(labels, pred)
    loss_1 = compute_loss_obj(labels, pred)
    
    return loss_0 + loss_1 


#def compute_loss_no_obj(labels, pred):
#    num = NUM_ANCHOR*(1+4+NUM_CLASSES)
#    var = labels.shape[1]
#    for i in range(0,var,num):    
#           if():
#    
#    
#    
#def compute_loss_obj(labels, pred):
#    #use tf.slice


#@tf.function
#def train_step(gen_x, dis_real_x, dis_y,gen_y): 
#  with tf.GradientTape() as tape:
#
#
#  return loss_gen+loss_dis
#
#
##### training ########
#for iter in range(num_training_iterations):
#	gen_x = get_Gen_inp(GenInputLenght,batch_size)
#	dis_real_x = get_Dist_real(DisInputLength,batch_size)
#
#	#other half of disc input is gen output
#	dis_y = np.concatenate([np.ones(batch_size),np.zeros(batch_size)])
#	gen_y = np.ones(batch_size)
#
#
#	loss = train_step(tf.convert_to_tensor(gen_x,dtype=tf.float32), tf.convert_to_tensor(dis_real_x,dtype=tf.float32),\
#	tf.convert_to_tensor(dis_y,dtype=tf.float32), tf.convert_to_tensor(gen_y,dtype=tf.float32))
#
#
#	print(loss.numpy().mean())








'''
	# Update the model with the changed weights!
	if iter % 100 == 0:     
		Gen.save_weights(checkpoint_prefix_gen)
		Dis.save_weights(checkpoint_prefix_dis)
			
    
# Save the trained model and the weights
Gen.save_weights(checkpoint_prefix_gen)
Dis.save_weights(checkpoint_prefix_dis)







data = signal.triang(28)
print(np.shape(data))
data_shift = (np.random.normal(0,10,100)).astype(int)

get_batch(batch_size= 10,seq_len = 20)

fig = plt.figure(figsize=(1,2))
sub = fig.add_subplot(1,2,1)
sub1 = fig.add_subplot(1,2,2)
sub1.plot(data)
sub.plot(np.roll(data,-4))
sub.set_title('X1')
sub1.set_title('X2')
plt.show()
'''

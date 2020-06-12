#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:53:37 2020

@author: maniac
"""

from hyper_parameters import IMAGE_H,IMAGE_W,GRID_DIM,NUM_ANCHOR,NUM_CLASSES,\
SHUFFLE_BUFFER_SIZE,feature_description 
from inference import decode_yolo_output,idx2cat,display_yolo_output
import tensorflow as tf
import numpy as np
import cv2



ANCHORS = tf.constant([[0.4,0.2],[0.3,0.3],[0.2,0.4]])
#aspect ratio of anchors [2,1,0.5] x/y which is w/h

def load_img_tf(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img #image tensor

def pre_process(annotation):
    feature_dict = tf.io.parse_single_example(annotation,feature_description)
    
    ##IMAGE_PREPROCESSING
    img = tf.io.decode_jpeg((feature_dict['image/encoded']))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_W, IMAGE_H])
    img_width = feature_dict['image/width']
    img_height = feature_dict['image/height']
    
    ##LABEL_PREPROCESSING
    categories = tf.one_hot(tf.sparse.to_dense(feature_dict['image/object/class/label']),NUM_CLASSES)
    
    no_of_objects = tf.shape(categories)[0]
    
    #write output y_hat as output of our neural net
    boxes = tf.stack([
    tf.sparse.to_dense(feature_dict['image/object/bbox/xmin']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/xmax']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/ymin']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/ymax'])]
    )
    width = boxes[1]-boxes[0]
    height = boxes[3] - boxes[2]
    centre_x = boxes[0] + width/2
    centre_y =boxes[2] + height/2
   
    x_scale = tf.cast(tf.math.divide(IMAGE_W,img_width),tf.float32)
    y_scale =  tf.cast(tf.math.divide(IMAGE_H,img_height),tf.float32)
    
    width=   tf.cast(width * x_scale,tf.int32)
    height = tf.cast(height * y_scale,tf.int32)
    centre_x = tf.cast(centre_x * x_scale,tf.int32)
    centre_y = tf.cast(centre_y * y_scale,tf.int32)
    
    #GRID x GRID x NUM_ANCHORS x (5+NUM_CATEG)
    
#    img1 = img.numpy()*255
#    img1 = np.ndarray.astype(img1,np.uint8)
#    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
##    print(parsed_example['image/filename'])
#    cv2.imshow('image',img1)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    
    ratio_mat = tf.linalg.diag([2,1,0.5]) #anchor w/h ratios
    mat = tf.ones([NUM_ANCHOR,no_of_objects],tf.float32)
    mat = tf.math.abs(tf.matmul(ratio_mat,mat) - tf.cast((height/width),tf.float32))
    mat = tf.one_hot(tf.argmin(mat),NUM_ANCHOR)
    anchor_loc = tf.matmul( [tf.range(NUM_ANCHOR)],tf.cast(tf.transpose(mat),tf.int32) )
    anchor_loc = tf.cast(anchor_loc,tf.int32)
    anchor_loc = tf.reshape(anchor_loc,[-1])    
    
    Factor_x = tf.cast(tf.math.divide(IMAGE_W,GRID_DIM),tf.int32)
    Factor_y = tf.cast(tf.math.divide(IMAGE_H,GRID_DIM),tf.int32)
    Num_x = tf.cast(tf.math.divide(centre_x,Factor_x),tf.int32 )
    Num_y = tf.cast(tf.math.divide(centre_y,Factor_y),tf.int32 )
    #num_x(img_w) is col no and num_y(img_h) is row no

    #No more required now INDEX = Num_y*GRID_DIM+self.Num_x
    
    #convert to yolov3 format cx cy
    sigmoid_tx = (centre_x  % Factor_x)/Factor_x
    sigmoid_ty = (centre_y  % Factor_y)/Factor_y

    Y_HAT = tf.zeros([GRID_DIM,GRID_DIM,NUM_ANCHOR,5+NUM_CLASSES])

#WE ONLY WANT ONE BOUNDING BOX PREDICTOR PER OBJECT
#IOU cannot be part of LOSS Function since it is not diffrenciable      
    for i in tf.range(no_of_objects):   #IN future I might remove this loop for something fancy  
        w = (tf.gather(width,i) /IMAGE_W)
        h= (tf.gather(height,i) /IMAGE_H)

       
        tw = tf.math.log( tf.cast(w,tf.float32)/ANCHORS[anchor_loc[i],0])
        th = tf.math.log( tf.cast(h,tf.float32)/ANCHORS[anchor_loc[i],1] )
        y_= tf.stack(  [tf.constant(1.0),tf.cast(sigmoid_tx[i],tf.float32),tf.cast(sigmoid_ty[i],tf.float32),tw,th],0)  
        obj_box = tf.concat([y_,categories[i]],0)
#                            #objectness,cx,cy,tw,th,cat
#        
        Y_HAT = tf.tensor_scatter_nd_update(Y_HAT, [[Num_y[i],Num_x[i],anchor_loc[i]]], [obj_box] )
#        #row,column,anchor,box
#        #check if extra square bracets are used in tensor_scatter

    return img, Y_HAT



def create_dataset(): 
    
    
    dataset = tf.data.TFRecordDataset(['./train_record1_of_2.tfrecords'])
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
#    for item in dataset.take(1):
#        pre_process(item)
    dataset = dataset.map(pre_process,num_parallel_calls = 4)#use 4 threads
    dataset = dataset.batch(2)

    for item in dataset.take(1):
        img = item[0][0]
        y = item[1][0]
        
        
        img1 = img.numpy()*255
        img1 = np.ndarray.astype(img1,np.uint8)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #    print(parsed_example['image/filename'])
        cv2.imshow('image',img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#        print(y.shape,img.shape)
         #list of boxes and cat names
        boxes,ids = decode_yolo_output(y)
#        print("hii",boxes)
        display_yolo_output(img,y) #display results

    return dataset.prefetch(1)



#import cv2
#for item in dataset.take(1):
#    parsed_example = tf.io.parse_single_example(item,hyper_parameters.feature_description)
#    img = tf.io.decode_jpeg((parsed_example['image/encoded']))
#    #Tensorflow image format is rgb so make sure during testing
#    #if you are loading images by opencv It uses bgr
#    img = img.numpy()
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    print(parsed_example['image/filename'])
#    cv2.imshow('image',img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

#Now dataset is created 
#use map to set image and label items
#learn label rep from tutorial first then implement your own
#use tf functions in all process
#create label for one scale as of now
#write functions to undo the oprations and watch true reshaped image 
#also draw the boxes 
    
def main():
    create_dataset()   

if __name__== "__main__":
    main()
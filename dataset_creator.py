#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:53:37 2020

@author: maniac
"""
import hyper_parameters
import tensorflow as tf
Anchors = tf.constant([[0.4,0.2],[0.2,0.4]])

def load_img_tf(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img #image tensor

def pre_process(annotation,Anchors):
    feature_dict = tf.io.parse_single_example(annotation,hyper_parameters.feature_description)
    
    ##IMAGE_PREPROCESSING
    img = tf.io.decode_jpeg((feature_dict['image/encoded']))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [hyper_parameters.IMAGE_W, hyper_parameters.IMAGE_H])
    img_width = feature_dict['image/width']
    img_height = feature_dict['image/height']
    
    ##LABEL_PREPROCESSING
    categories = tf.one_hot(tf.sparse.to_dense(feature_dict['image/object/class/label']),hyper_parameters.NUM_CLASSES)
    
    no_of_objects = tf.shape(categories)[0]
    
    #write output y_hat as output of our neural net
    boxes = tf.stack(
    tf.sparse.to_dense(feature_dict['image/object/bbox/xmin']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/xmax']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/ymin']),
    tf.sparse.to_dense(feature_dict['image/object/bbox/ymax'])
    )
    
    #GRID x GRID x NUM_ANCHORS x (5+NUM_CATEG)
    
        
    #lets find grids corresponding to all boxes
    
    #scale box coordinates since image is scaled above as per preprocessing
    x_scale = tf.math.divide(IMAGE_W,img_width)
    y_scale =  tf.math.divide(IMAGE_H,img_height)
    xmin = tf.cast(tf.math.round(tf.math.multiply( boxes[0,:] , x_scale)),dtype = tf.int32)
    ymin = tf.cast(tf.math.round(tf.math.multiply( boxes[2,:] , y_scale)),dtype =tf.int32)
    xmax = tf.cast(tf.math.round(tf.math.multiply( boxes[1,:] , x_scale)),tf.int32)
    ymax = tf.cast(tf.math.round(tf.math.multiply( boxes[3,:] , x_scale)),tf.int32)
    #find important box info
    width = xmax- xmin
    height = ymax- ymin
    centre_x = xmin + tf.math.divide(width,2)
    centre_y = xmax + tf.math.divide(height,2)
    
    Factor_x = tf.math.divide(IMAGE_W,GRID_DIM)
    Factor_y = tf.math.divide(IMAGE_H,GRID_DIM)
    Num_x = tf.cast(tf.math.divide(centre_x,Factor_x),tf.int32 )
    Num_y = tf.cast(tf.math.divide(centre_y,Factor_y),tf.int32 )
    INDEX = Num_y*GRID_DIM+Num_x
    unique = tf.unique(INDEX)
    
    for i in tf.range(no_of_objects):
            #scale the bounding box as per input_scaling        
            if(INDEX[i] in unique):
                #all anchors point to same box
        sigma_tx = (centre_x  %Factor_x)/Factor_x
        sigma_ty = (centre_y  % Factor_y)/Factor_y
        
        
        #tensorflow images are loaded as (ht,width,ch)
        if (width<height):#use anchor 1
            tw = tf.math.log( (width /IMAGE_W) /Anchors[1,0])
            th = tf.math.log( (height/IMAGE_H) /Anchors[1,1] )
            YOLO_BOXES[Num_x,Num_y,1,0] = 1
        else:
            tw = tf.math.log( (width /IMAGE_W) /Anchors[0,0])
            th = tf.math.log( (height/IMAGE_H) /Anchors[0,1] )
            YOLO_BOXES[Num_x,Num_y,0,0] = 1
         #objectness 
    
    
    
    return img

def create_dataset():
    
    
    dataset = tf.data.TFRecordDataset(['./train_record1_of_2.tfrecords'])
    dataset = dataset.shuffle(hyper_parameters.SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(pre_process,num_parallel_calls = 4)#use 4 threads
    dataset = dataset.batch(2)
    return dataset.prefetch(1)


import cv2
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
    pass    

if __name__== "__main__":
    main()
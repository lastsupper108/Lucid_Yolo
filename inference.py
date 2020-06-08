#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 01:27:40 2020

@author: maniac
"""

from hyper_parameters import OBJECTNESS_THREHOLD,GRID_DIM,NUM_ANCHOR,\
IMAGE_H,IMAGE_W,NUM_CLASSES,get_dictionaries
import tensorflow as tf
ANCHORS = tf.constant([[0.4,0.2],[0.3,0.3],[0.2,0.4]])
idx2cat,cat2idx =  get_dictionaries()

import cv2
import numpy as np

#input is single prediction y
#output is boxes,cat list
def decode_yolo_output(y):

    #THIS FUNCTION works for SIG(TX),SIG(TY) values in y NOT TX,TY

    objectness= tf.reshape(tf.cast( y[...,0] >OBJECTNESS_THREHOLD ,tf.int32),[-1])
    boxes_loc = tf.where(objectness)
    no_ofBoxes = tf.shape(boxes_loc)[0]
    index = tf.cast(boxes_loc/NUM_ANCHOR,tf.int64)
    #index consist of index of box with object in itp
    
    anchor_index = boxes_loc%NUM_ANCHOR
    Num_y = tf.cast(index/GRID_DIM,tf.int32)
    Num_x = index%GRID_DIM  
    
    sig_tx= tf.gather( tf.reshahape(y[...,1],[-1]) , boxes_loc)
    sig_ty= tf.gather( tf.reshape(y[...,2],[-1]) , boxes_loc)
    tw= tf.gather( tf.reshape(y[...,3],[-1]) , boxes_loc)
    th= tf.gather( tf.reshape(y[...,4],[-1]) , boxes_loc)
    
    
    Factor_x = tf.math.divide(IMAGE_W,GRID_DIM)
    Factor_y = tf.math.divide(IMAGE_H,GRID_DIM)
    
    #in pixels
    centre_x= Factor_x*(tf.cast(Num_x,tf.float32) + sig_tx)  
    centre_y= Factor_y*(tf.cast(Num_y,tf.float32) + sig_ty)
    
    out = tf.gather(ANCHORS,anchor_index)
    width = tf.matmul(tf.math.exp(tw),out[0]) #ANCHOR first dim is width
    height = tf.matmul(tf.math.exp(th),out[1])
    
    #in pixels
    width = width * Factor_x 
    height = height * Factor_y
    
    #Convert back to categories
    one_hot_vectors = tf.reshape(y[...,5:],[-1,NUM_CLASSES])
    class_ID = tf.argmax(one_hot_vectors, axis=1)
    
    
    x_min= tf.maximum((centre_x - width/2),0)
    x_max= tf.minimum((centre_x + width/2),IMAGE_W-1)
    y_min= tf.maximum((centre_y - height/2),0)
    y_max= tf.minimum((centre_y + height/2),IMAGE_H-1)
    boxes = tf.stack([x_min,y_min,x_max,y_max])
    
    return boxes,class_ID

#input is (image,y)
#image size is fixed as per hyper_parameters
def display_yolo_output(img,y): 
    boxes,class_ID = decode_yolo_output(y)
    class_ID = class_ID.numpy()
    box_labels = [idx2cat[i] for i in class_ID.numpy()]

    img =  img.numpy()*255
    img = np.ndarray.astype(img,np.uint8)
    
    boxes = boxes.numpy()
    
    for i in range(0, len(boxes[0])):
        # changed color and width to make it visible
        cv2.rectangle(img, (boxes[0][i], boxes[1][i]), (boxes[2][i], boxes[3][i]), (255, 0, 0), 2)
    cv2.imshow("img", img[...,::-1])#back convert to bgr from rgb to display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

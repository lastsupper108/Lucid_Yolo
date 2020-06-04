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
    YOLO_BOXES = tf.zeros(GRID_DIM,GRID_DIM,2,25)
        
    
    for i in tf.range(no_of_objects):
            #scale the bounding box as per input_scaling
    x_scale = tf.math.divide(IMAGE_W,img_width)
    y_scale =  tf.math.divide(IMAGE_H,img_width)

    xmin = int(np.round(box.x_min * x_scale))
    ymin = int(np.round(box.y_min * y_scale))
    xmax = int(np.round(box.x_max * x_scale))
    ymax = int(np.round(box.y_max * y_scale))
    b = Box([xmin,ymin,xmax,ymax])
    


    width = boxes[1,i]- boxes[0,i]
        height = boxes[3,i]- boxes[2,i]
        centre_x = boxes[0,i]+tf.math.divide(width,2)
        centre_y = boxes[2,i]+tf.math.divide(height,2)
        
        Factor_x = IMAGE_W/GRID_DIM
        Factor_y = IMAGE_H/GRID_DIM
        self.Num_x = int(self.centre_x/Factor_x) 
        self.Num_y = int(self.centre_y/Factor_y )        

        
    
    
    
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:06:31 2020

@author: maniac
"""

def binary_cross_entropy(logits, labels):
    epsilon = 1e-7
    logits = tf.clip_by_value(logits, epsilon, 1 - epsilon)
    return -(labels * tf.math.log(logits) +(1 - labels) * tf.math.log(1 - logits))

def calc_loss(y,y_hat):
    print(y.shape)
    sig_tx_ty = tf.sigmoid(y_pred[..., 0:2])
    tw_th = y_pred[..., 2:4]
    
    tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    
    
    #First we need a true mask to select certain values for which we 
    #want to backpropagateng 
    #when a bounding box has iou greater than threshold we will not 
    # calculate objectness loss for that box we will not punish it 
    #for showing object this will make network more generalised
    # refer this link https://stackoverflow.com/questions/56199478/what-is-the-purpose-of-ignore-thresh-and-truth-thresh-in-the-yolo-layers-in-yolo 
    
    
    #I have found a new hack to achieve the same since my anchors are all 
    #to centre of box. I will not punish grids in neighbourhood top left 
    #right and bottom my ground truth box. It will have similar effiect.
    #I will not do it for bordering grids i.e keep padding of 1 grid cell 
    #so that code will be simplified
    
    #objectness loss
    obj_entropy = binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
    #ignore mask is the one which saves objects from no obj loss punishment
    #since those boxes have some good iou
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

return obj_loss + noobj_loss


        # class loss
        class_loss = binary_cross_entropy(pred_class, true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
return class_losss



wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
return wh_loss


        xy_loss = true_obj * xy_loss * weight

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

return xy_loss

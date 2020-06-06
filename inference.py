#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 01:27:40 2020

@author: maniac
"""

#input is single prediction y
#output is boxes,cat list
def decode_yolo_output(y):
    boxes=[]
    categories = []
    return boxes,categories

#input is (image,y)
#image size is fixed as per hyper_parameters
def display_yolo_output(img,y):
    box,cat = decode_yolo_output(y)
    pass
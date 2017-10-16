# -*- coding: utf-8 -*-
"""
Created on Wed May 17 14:09:20 20
"""
import numpy as np

ratio_a = [1, 2,3, 1/2 , 1/3]


def box_scale(k):
    
    S_min = 0.2
    S_max = 0.9
    M = 5

    S_k = S_min + (S_max - S_min) * (k - 1.0) / (M - 1.0) 
    return S_k
    
    

def Default_boxes(layer_shape , L):   # Default_boxes for L_th feature map 

    layer_H = layer_shape[1]
    layer_W = layer_shape[0] 

    s_k = box_scale(L + 1)
   # s_k1 = box_scale(L + 2)
    
    x , y = np.mgrid[0:layer_H , 0:layer_W]
   
    
    box_w = np.zeros((len(ratio_a), ), dtype=np.float32)
    box_h = np.zeros((len(ratio_a), ), dtype=np.float32)
                   
    for i in range(len(ratio_a)):

        box_w[i] = s_k * np.sqrt(ratio_a[i])
        box_h[i] = s_k/ np.sqrt(ratio_a[i])
#
    c_x = (x.astype(np.float32) + 0.5) / float(layer_W)
    c_y = (y.astype(np.float32) + 0.5) / float(layer_H)
#
    c_x = np.expand_dims(c_x, axis=-1)
    c_y = np.expand_dims(c_y, axis=-1)
    
    
    #x_min = max (lower_bound,c_x - box_w /2)
    #x_max = min(1 , c_x + box_w/2)
    
    #y_min = max(c_y - box_h/2,0)
    #y_max = min(1, c_y + box_h/2)

    return c_x , c_y , box_w , box_h
                
                
                
#        boxes.append(layer_boxes)
#

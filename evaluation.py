# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 21:43:30 2017

"""
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imshow
import pandas as pd
from PIL import Image
from VGG_bn import VGG_bn
from VGG import VGG
from Boxes import Default_boxes
from loader2 import load_train_data,load_list
import cv2
#import time




#######################################################
prior_scaling=[0.1, 0.1, 0.2, 0.2]
output_size = [[38,38],[19,19],[10,10],[5,5],[3,3]]             


N_Default_Boxes = 5

N_Classes = 2         # with background class

N_Channels = 3        # grayscale->1

N_Class_Predict = N_Default_Boxes * N_Classes  # number of class predictions per feature map cell

N_Loc_Predict  = N_Default_Boxes * 4  # number of localization regression predictions per feature map cell

IMG_H=300
IMG_W=300

M=5
#########################################################

tf.reset_default_graph()  

X = tf.placeholder(tf.float32, [1 ,IMG_H, IMG_W, N_Channels], name='x')   # input image batches

                         
vgg=VGG_bn(X)

SSD_logits = []
SSD_P = []
SSD_Loc = []



with tf.variable_scope('SSD_OUT1'):
    
     W1= tf.get_variable('weight1' , [3,3, 512, N_Class_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     W2= tf.get_variable('weight2' , [3,3, 512, N_Loc_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     
     SSD_OUT_Predict = tf.nn.conv2d(vgg.conv4_3, W1, strides=[1,1,1,1] , padding ='SAME')
     SSD_OUT_Loc = tf.nn.conv2d(vgg.conv4_3, W2, strides=[1,1,1,1] , padding ='SAME')

     Cls_pred = tf.reshape(SSD_OUT_Predict, SSD_OUT_Predict.get_shape().as_list()[0:3]+[M, N_Classes])
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3]+[M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)
         
with tf.variable_scope('SSD_OUT2'):

     W1= tf.get_variable('weight1' , [3,3, 1024, N_Class_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     W2= tf.get_variable('weight2' , [3,3, 1024, N_Loc_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     
     SSD_OUT_Predict = tf.nn.conv2d(vgg.convSSD2, W1, strides=[1,1,1,1] , padding ='SAME')
     SSD_OUT_Loc = tf.nn.conv2d(vgg.convSSD2, W2, strides=[1,1,1,1] , padding ='SAME')

     Cls_pred = tf.reshape(SSD_OUT_Predict, SSD_OUT_Predict.get_shape().as_list()[0:3]+[M, N_Classes])
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3]+[M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)

with tf.variable_scope('SSD_OUT3'):
    
     W1= tf.get_variable('weight1' , [3,3, 512, N_Class_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     W2= tf.get_variable('weight2' , [3,3, 512, N_Loc_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     
     SSD_OUT_Predict = tf.nn.conv2d(vgg.convSSD4, W1, strides=[1,1,1,1] , padding ='SAME')
     SSD_OUT_Loc = tf.nn.conv2d(vgg.convSSD4, W2, strides=[1,1,1,1] , padding ='SAME')
     
     Cls_pred = tf.reshape(SSD_OUT_Predict, SSD_OUT_Predict.get_shape().as_list()[0:3]+[M, N_Classes])
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3]+[M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)
     
with tf.variable_scope('SSD_OUT4'):
    
     W1= tf.get_variable('weight1' , [3,3, 256, N_Class_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     W2= tf.get_variable('weight2' , [3,3, 256, N_Loc_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     
     SSD_OUT_Predict = tf.nn.conv2d(vgg.convSSD6, W1, strides=[1,1,1,1] , padding ='SAME')
     SSD_OUT_Loc = tf.nn.conv2d(vgg.convSSD6, W2, strides=[1,1,1,1] , padding ='SAME')
     
     Cls_pred = tf.reshape(SSD_OUT_Predict, SSD_OUT_Predict.get_shape().as_list()[0:3]+[M, N_Classes])
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3]+[M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)
     
     
with tf.variable_scope('SSD_OUT5'):
    
     W1= tf.get_variable('weight1' , [3,3, 256, N_Class_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     W2= tf.get_variable('weight2' , [3,3, 256, N_Loc_Predict], initializer= tf.truncated_normal_initializer(stddev=0.01))
     
     SSD_OUT_Predict = tf.nn.conv2d(vgg.convSSD8, W1, strides=[1,1,1,1] , padding ='SAME')
     SSD_OUT_Loc = tf.nn.conv2d(vgg.convSSD8, W2, strides=[1,1,1,1] , padding ='SAME')
     
     Cls_pred = tf.reshape(SSD_OUT_Predict, SSD_OUT_Predict.get_shape().as_list()[0:3]+[M, N_Classes])
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3]+[M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)
              
Flatten_logits=[]
Flatten_Loc=[]
Flatten_P=[]
 
     
for P in SSD_P:
    
    SSD_F_P= tf.reshape(P,[-1,N_Classes])
    Flatten_P.append(SSD_F_P)
    
for P in SSD_logits:
    
    SSD_F_logitcs = tf.reshape(P, [-1, N_Classes])
    Flatten_logits.append(SSD_F_logitcs)    
    

for i, P in enumerate(SSD_Loc):
    
    SSD_F_Loc= tf.reshape(P,[-1, 4])
    Flatten_Loc.append(SSD_F_Loc)
    
Pred_label =  tf.concat(Flatten_P,0)  
Pred_Loc =  tf.concat(Flatten_Loc,0) 
Pred_logits =  tf.concat(Flatten_logits,0) 
     
 

Default_Boxes=[]
Default_Anchor=[]
def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.
    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]



def pad_axis(x, offset, size, axis=0, name=None):
    """Pad a tensor on an axis, with a given offset and output size.
    The tensor is padded with zero (i.e. CONSTANT mode). Note that the if the
    `size` is smaller than existing size + `offset`, the output tensor
    was the latter dimension.
    Args:
      x: Tensor to pad;
      offset: Offset to add on the dimension chosen;
      size: Final size of the dimension.
    Return:
      Padded tensor whose dimension on `axis` is `size`, or greater if
      the input vector was larger.
    """
    shape = get_shape(x)
    rank = len(shape)
        # Padding description.
    new_size = tf.maximum(size-offset-shape[axis], 0)
    pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
    pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
    paddings = tf.stack([pad1, pad2], axis=1)
    x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
    shape[axis] = size
    x = tf.reshape(x, tf.stack(shape))
    return x




for i in range(5):
    
    layer_Shape= output_size[i]
    
    r_x , r_y , r_w , r_h = Default_boxes( layer_Shape , i)
    
    Default_Boxes.append([r_x , r_y , r_w , r_h])
    
    xmin = r_x - r_w/2
    xmax = r_x + r_w/2
    ymin = r_y - r_h/2
    ymax = r_y + r_h/2

    Default_Anchor.append([xmin , ymin , xmax , ymax])





def bboxes_decode(SSD_Loc, SSD_Pred, Default_Boxes):
           
    Dboxes = []  
    l_scores = []
    l_bboxes = []  

                                
    for i in range(5):
        
        r_x, r_y, r_w, r_h = Default_Boxes[i]

                                         # Compute center, height and width
        cx = SSD_Loc[i][:, :, :, :, 0] * r_w * prior_scaling[0] + r_x
        cy = SSD_Loc[i][:, :, :, :, 1] * r_h *prior_scaling[1] + r_y
        w = r_w * tf.exp(SSD_Loc[i][:, :, :, :, 2] *prior_scaling[2] )
        h = r_h * tf.exp(SSD_Loc[i][:, :, :, :, 3] *prior_scaling[3] )
    # Boxes coordinates.
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        Dbox = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
        
        Dboxes.append(Dbox)
        
    lp=[]           
    for i in range(5):
        
        p_shape = SSD_P[i].get_shape().as_list()
        predictions_layer = tf.reshape(SSD_P[i], tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = Dboxes[i].get_shape().as_list()
        localizations_layer = tf.reshape(Dboxes[i],tf.stack([l_shape[0], -1, l_shape[-1]]))
        lp.append(predictions_layer)
       
        ddd_scores = {}
        ddd_bboxes = {}

        for c in range(0, N_Classes):
            if c != 0:
                # Remove boxes under the threshold.
                scores = predictions_layer[:, :, c]
                fmask = tf.cast(tf.greater_equal(scores, 0.01), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                # Append to dictionary.
                ddd_scores[c] = scores
                ddd_bboxes[c] = bboxes

        l_scores.append(ddd_scores)
        l_bboxes.append(ddd_bboxes)
        # Concat results.
    
    d_scores = {}
    d_bboxes = {}
    for c in l_scores[0].keys():
        ls = [s[c] for s in l_scores]
        lb = [b[c] for b in l_bboxes]
        d_scores[c] = tf.concat(ls, axis=1)
        d_bboxes[c] = tf.concat(lb, axis=1)
    #return d_scores,d_bboxes
##########################################
    dd_scores = {}
    dd_bboxes = {}  
    for c in d_scores.keys():   
#            
        scores, idxes = tf.nn.top_k(d_scores[c], k=400, sorted=False)
#
#
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]

        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [d_bboxes[c], idxes],
                      dtype=[d_bboxes[c].dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]  
        dd_scores[c] = scores
        dd_bboxes[c] = bboxes
    

    r_scores = {}
    r_bboxes = {}

    def bboxes_nms(score,bbox,nms_threshold=0.2, keep_top_k=200):
        idxes = tf.image.non_max_suppression(bbox, score, keep_top_k, nms_threshold)
        scores = tf.gather(score, idxes)
        bboxes = tf.gather(bbox, idxes)
        scores = pad_axis(scores, 0, 200, axis=0)
        bboxes = pad_axis(bboxes, 0, 200, axis=0)
        return scores,bboxes
        

    for c in dd_scores.keys(): 
#        

        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1], 0.1, 200) ,
                      (dd_scores[c], dd_bboxes[c]),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
#        
        r_scores[c]=scores
        r_bboxes[c]=bboxes
    return r_scores,r_bboxes



r_scores,r_bboxes= bboxes_decode(SSD_Loc, SSD_P, Default_Boxes)

saver = tf.train.Saver()
        
with tf.Session() as sess:
     saver.restore(sess, "c:/newgraph/model16.ckpt")             

     image_list = load_list()

     Batch_size = 1
     start=75
     DF, train_img = load_train_data(image_list,start,Batch_size)
    
     image = np.expand_dims(train_img[0],0)
        
     sc,boxd= sess.run([r_scores,r_bboxes], feed_dict = {X:image}) 

cv2.namedWindow("image")
cv2.imshow("image",train_img[0])
cv2.waitKey(100) 

transp=np.transpose(np.nonzero(sc[1]))

for index in range(len(transp)):
    row,col = transp[index]
    c=boxd[1][0][col]
    x1=int(np.round(np.maximum(0,c[0]*299)))
    y1=int(np.round(np.maximum(0,c[1]*299)))
    x2=int(np.round(np.minimum(299,c[2]*299)))
    y2=int(np.round(np.minimum(299,c[3]*299)))
    
    cv2.rectangle( train_img[0] , ( x1,y1),(x2, y2),(255,0,255),5,5)

cv2.waitKey()                           
           
         
         

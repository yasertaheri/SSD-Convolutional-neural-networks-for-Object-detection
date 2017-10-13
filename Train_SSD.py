

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imshow
import pandas as pd
from PIL import Image
from VGG import VGG
from VGG_bn import VGG_bn
from Boxes import Default_boxes
from loader2 import load_list, load_train_data,Bach_Next
#import cv2
#import time

######################################################################################################################################
######################################################### Setting ####################################################################             
             
# Offset is relative to upper-left-corner and lower-right-corner of the feature map cell

N_Default_Boxes = 5

N_Classes = 2         # Class of class + 1 background class

N_Channels = 3

N_Class_Predict = N_Default_Boxes * N_Classes  # number of class predictions per feature map cell

N_Loc_Predict  = N_Default_Boxes * 4  # number of localization regression predictions per feature map cell

M=5
######################################################################################################################################
######################################################################################################################################
# Bounding box parameters

BOX_THRESH = 0.5         # match ground-truth box to default boxes exceeding this IOU threshold, during data prep

IMG_H, IMG_W = 300, 300

LOC_LOSS_WEIGHT = 1.  # weight of localization loss: loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss

NUM_EPOCH = 10000
  
BATCH_SIZE = 4  # batch size for training (relatively small)

S=9695

####################################################################################################################################
####################################################################################################################################
###################################################################################################################################

tf.reset_default_graph()  

X = tf.placeholder(tf.float32, [BATCH_SIZE ,IMG_H, IMG_W, N_Channels], name='x')   # input image batches

Ycls = tf.placeholder(tf.float32, [S*BATCH_SIZE, ], name='y')   # input image batches
Yloc = tf.placeholder(tf.float32, [S*BATCH_SIZE, 4], name='y')  # input image batches
Ysc = tf.placeholder(tf.float32, [S*BATCH_SIZE, ], name='y')   # input image batches

                         
vgg=VGG_bn(X)
    
#######################################################################################################################################
#######################################################################################################################################

SSD_logits = []
SSD_Loc = []
SSD_P=[]



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
     Loc_pred = tf.reshape(SSD_OUT_Loc, SSD_OUT_Loc.get_shape().as_list()[0:3] + [M, 4])
     
     p=tf.nn.softmax(Cls_pred)
     SSD_P.append(p)
     SSD_logits.append(Cls_pred)
     SSD_Loc.append(Loc_pred)
     


Flatten_logits1=[]
Flatten_Loc1=[]
Flatten_P1=[]

for i in range(BATCH_SIZE):
    
    Flatten_logits=[]
    Flatten_Loc=[]
    Flatten_P=[]
 
    for P in range(len(SSD_P)):
    
        SSD_F_P= tf.reshape(SSD_P[P][i],[-1, N_Classes])
        SSD_F_logitcs = tf.reshape(SSD_logits[P][i], [-1, N_Classes])
        SSD_F_Loc= tf.reshape(SSD_Loc[P][i],[-1, 4])

        
        Flatten_P.append(SSD_F_P)       
        Flatten_logits.append(SSD_F_logitcs)    
        Flatten_Loc.append(SSD_F_Loc)

    
    Pred_Loc =  tf.concat(Flatten_Loc,0) 
    Pred_logits =  tf.concat(Flatten_logits,0)  
    Pred_P =  tf.concat(Flatten_P,0)
    
    Flatten_logits1.append(Pred_logits)
    Flatten_Loc1.append(Pred_Loc)
    Flatten_P1.append(Pred_P)

Pred_Loc =  tf.concat(Flatten_Loc1,0) 
Pred_logits =  tf.concat(Flatten_logits1,0)  
Pred_P =  tf.concat(Flatten_P1,0)  

#######################################################################################################################################

def smooth(X):
    return  tf.where(X < 1, 0.5*X**2 , tf.abs(X)-0.5)
              
########################################################################################################################################

def loss (gclasses, glocalisations, gscores ):    
    
   
                
    pmask = gscores > 0.5
    fpmask = tf.cast(pmask, tf.float32)
    n_positives = tf.reduce_sum(fpmask)

    # Hard negative mining...
    no_classes = tf.cast(pmask, tf.int32)
       
    
    nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
    fnmask = tf.cast(nmask, tf.float32)
    nvalues = tf.where(nmask,tf.cast(Pred_P[:,0],tf.float32), 1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(3 * n_positives, tf.int32)
    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
    max_hard_pred = -val[-1]
        # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, tf.float32) 
        
        
    loss_p = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Pred_logits, labels=tf.cast(gclasses, tf.int32))
    loss_p = (tf.reduce_sum(loss_p * fpmask))

    loss_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Pred_logits,labels=no_classes)
    loss_n = (tf.reduce_sum(loss_n * fnmask))
       
    weights = tf.expand_dims(1 * fpmask, axis=-1)
    loss_ll = (smooth(Pred_Loc - tf.cast(glocalisations,tf.float32)))
    loss_l = tf.div(tf.reduce_sum(loss_ll * weights), 1, name='value')


    lo =  (loss_l + loss_p + loss_n)/n_positives
    
    return lo
         
L = loss(Ycls, Yloc, Ysc)       
   
Optimizer= tf.train.AdamOptimizer(0.0001). minimize(L)  
                
init=  tf.global_variables_initializer()               
################################################################################################
saver = tf.train.Saver()

image_list = load_list()

with tf.Session() as sess:
     sess.run(init)

     c=np.load('C:/Users/y_moh/Desktop/vgg16_weights.npz')
     p=sorted(c.keys())    
     for i, k in enumerate(p):
         if i<26:
            sess.run(vgg.parameters[i].assign(c[k]))    

     
     #saver.restore(sess, "c:/newgraph/model17.ckpt")                                    
     for epoch in range(NUM_EPOCH):
         
         
         for Start in range(0,len(image_list)-4,BATCH_SIZE):
             
             gclasses, glocalisations, gscores, image  = Bach_Next(image_list, Start,BATCH_SIZE)

             _,l= sess.run([Optimizer,L], feed_dict = {X:image, Ycls: gclasses, Yloc:glocalisations , Ysc:gscores  })
             print(epoch,"   ",Start,"  ",l) 

         
         save_path = saver.save(sess, "c:/newgraph/model20.ckpt")             
                    

           
        
        



        


    

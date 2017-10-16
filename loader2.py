
import scipy
import numpy as np
from VGG import VGG
import os
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
from scipy.misc import imread, imresize, imshow
from PIL import Image
from Boxes import Default_boxes

######################################################################################################################################
######################################################### Setting ####################################################################             
output_size = [[38,38],[19,19],[10,10],[5,5],[3,3]]             
prior_scaling=[0.1, 0.1, 0.2, 0.2]

N_Default_Boxes = 5

N_Classes = 2         # 2 signs + 1 background class

N_Channels = 3        # grayscale->1

dir = 'C:\VOCdevkit\VOC2012'

Annotation = os.path.join(dir,'Annotations')
Images = os.path.join(dir,'JPEGImages')
set_dir = os.path.join(dir, 'ImageSets', 'Main')

   
list_labels=[ 'aeroplane', 'bicycle', 'bird', 'boat',
              'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse',
              'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train','tvmonitor']
              
list_labels2=['person']              
              
              
def load_img(img_filename):
    return scipy.misc.imread(os.path.join(Images, img_filename + '.jpg'))              
    
    
def imgs_from_category(cat_name, dataset):
    filename = os.path.join(set_dir, cat_name + "_" + dataset + ".txt")
    df = pd.read_csv(filename, delim_whitespace=True, header=None, names=['filename', 'true'])
    return df

def imgs_from_category_as_list(cat_name, dataset):
    df = imgs_from_category(cat_name, dataset)
    df = df[df['true'] == 1]
    return df['filename'].values    

def annotation_file_from_img(img_name):
    return os.path.join(Annotation, img_name) + '.xml'
    

def load_annotation(img_filename):
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml,"lxml") 

def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)
    

        
        
#train_img_list = imgs_from_category_as_list(cat, 'train')


#a = load_annotation(train_img_list[0])


def load_list():
    
    train_img_list=[]
    for C in list_labels2:           
        train_img_list.append(imgs_from_category_as_list(C, 'train'))
        #train_img_list.append(imgs_from_category_as_list(C, 'trainval'))
    train_img_list= np.hstack(train_img_list) 
    return train_img_list

def load_train_data(train_img_list, im , batch_size):

    y= {}
    image_list=[]
    for i,item in enumerate(train_img_list[im:im+batch_size]):
        
        image=load_img(item)
        
        h=image.shape[0]
        w=image.shape[1]
        
        image = scipy.misc.imresize(image, [300,300], interp='bilinear', mode=None)
                
        image_list.append(image)
        

        anno = load_annotation(item)
        objs = anno.findAll('object')
        data = []

        for obj in objs:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                               
                #fname = anno.findChild('filename').contents[0]
                if name_tag.contents[0] in list_labels2:
                   cls=list_labels2.index(name_tag.contents[0]) + 1
                else:
                   continue
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])/w
                ymin = int(bbox.findChildren('ymin')[0].contents[0])/h
                xmax = int(bbox.findChildren('xmax')[0].contents[0])/w
                ymax = int(bbox.findChildren('ymax')[0].contents[0])/h
                data.append([cls, xmin, ymin, xmax, ymax])
        data= np.vstack(data)       
        y[i] = {'GT':data}
           
    return y, image_list

    
Default_Boxes=[]
Default_Anchor=[]

for i in range(5):
    
    layer_Shape= output_size[i]
    r_x , r_y , r_w , r_h = Default_boxes( layer_Shape , i)
    
    Default_Boxes.append([r_x , r_y , r_w , r_h])
    
    xmin = r_x - r_w/2
    xmax = r_x + r_w/2
    ymin = r_y - r_h/2
    ymax = r_y + r_h/2

    Default_Anchor.append([xmin , ymin , xmax , ymax])
        
    

def calc_iou(box_a, box_b):
	
	
    #Calculate the Intersection Over Union of two boxes
#	Each box specified by upper left corner and lower right corner:
#	(x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner
#	Returns IOU value
      
	

      x_overlap = np.maximum(0.0, np.minimum(box_a[2], box_b[2]) - np.maximum(box_a[0], box_b[0]))
      y_overlap = np.maximum(0.0, np.minimum(box_a[3], box_b[3]) - np.maximum(box_a[1], box_b[1]))
      intersection = x_overlap *y_overlap

      
	# Calculate union
      area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
      area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

      union = area_box_a + area_box_b - intersection

      iou = intersection / union

      return iou

#############################################################################################################3




def pre_process (Gboxes):
    
    layers_class=[]
    layers_loc=[]
    layers_score=[]
    
    for i in range(5):
        
        Y_cls = np.zeros_like(Default_Anchor[i][0])
        Y_Score = np.zeros_like(Default_Anchor[i][0])

        X_min = np.zeros_like(Default_Anchor[i][0])
        Y_min = np.zeros_like(Default_Anchor[i][0])
        X_max = np.ones_like(Default_Anchor[i][0])
        Y_max = np.ones_like(Default_Anchor[i][0])

        
        N = Gboxes.shape[0]
        
        
    
        for idx in range(N):
        
            
            Target_box=np.stack([Gboxes[idx][1], Gboxes[idx][2], Gboxes[idx][3], Gboxes[idx][4]])
            io=calc_iou(Default_Anchor[i], Target_box)
            
            mask = io > 0.5
            fmask = mask.astype(np.float32)
            #mask =  tf.where(io <0 , 0, 1 )
    
            Y_Score = np.where(mask, io, Y_Score)

            Y_cls = (1 - fmask) * Y_cls  + fmask * Gboxes[idx][0]

            X_min =(1 - fmask) * X_min + fmask * Gboxes[idx][1]
            Y_min =(1 - fmask) * Y_min + fmask * Gboxes[idx][2]
            X_max =(1 - fmask) * X_max + fmask * Gboxes[idx][3]
            Y_max =(1 - fmask) * Y_max + fmask * Gboxes[idx][4]
            
                    
                     
            #mask = np.where(io > 0.5 , 1 , 0 )
            
        c_x= (X_min + X_max)/2
        c_y= (Y_min + Y_max)/2
        w = X_max - X_min
        h = Y_max - Y_min
        

        c_x = (c_x - Default_Boxes[i][0])/ Default_Boxes[i][2]/prior_scaling[0]
        c_y = (c_y - Default_Boxes[i][1])/ Default_Boxes[i][3]/prior_scaling[1]
        w = np.log(w / Default_Boxes[i][2])/prior_scaling[2]
        h = np.log(h / Default_Boxes[i][3])/prior_scaling[3]
        

        
        Y_loc = np.stack([c_x,c_y,w,h],-1) 
        
        layers_class.append( np.reshape(Y_cls, [-1]))
        layers_score.append(np.reshape(Y_Score, [-1]))
        layers_loc.append(np.reshape(Y_loc, [-1,4]))

    
    gclasses = np.concatenate(layers_class, axis=0)
    gscores = np.concatenate(layers_score, axis=0)
    glocalisations = np.concatenate(layers_loc, axis=0)
    return gclasses, gscores, glocalisations

##############################################################################################################       


def Bach_Next(list_image, im,Batch_size):
    DF, train_img = load_train_data(list_image,im,Batch_size)
    gcls=[]
    gloc=[]
    gsc=[]
    
    for i  in range(0,Batch_size,1):
        Gboxes = DF[i]['GT']
        gclasses, gscores, glocalisations = pre_process (Gboxes)       
        gcls.append(gclasses)
        gloc.append(glocalisations)
        gsc.append(gscores)
    gc=np.concatenate(gcls,0)
    gl=np.concatenate(gloc,0)
    gs=np.concatenate(gsc,0)
    return gc,gl,gs,train_img


       

#for a in list_labels: 
#image_originals, DF =preprocess_data(cc)



  

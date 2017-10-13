import tensorflow as tf


class VGG:
    
    def __init__(self,imgs):
        self.imgs = imgs
        self.convlayers()
        
        
    def convlayers(self):
        
        self.parameters = []
        self.SSD_outputs = []

 # zero-mean input
        with tf.variable_scope('preprocess'):
             mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1, 3], name='img_mean')
             image = self.imgs - mean  
             
        with tf.variable_scope('conv1_1'):
             W = tf.get_variable('weight', shape = [3,3,3,64], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [64], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(image,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv1_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv1_1/W",W)
             
        with tf.variable_scope('conv1_2'):
             W = tf.get_variable('weight', shape = [3,3,64,64], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [64], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv1_1 ,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv1_2 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv1_2/W",W)
                          

             
        with tf.name_scope('pool1'):
             self.pool1= tf.nn.max_pool(self.conv1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool1")
             
             

        with tf.variable_scope('conv2_1'):
             W = tf.get_variable('weight', shape = [3,3,64,128], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool1,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv2_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv2_1/W",W)
             
        with tf.variable_scope('conv2_2'):
             W = tf.get_variable('weight', shape = [3,3,128,128], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv2_1,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv2_2 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv2_2/W",W)
             
        with tf.name_scope('pool2'):
            self.pool2= tf.nn.max_pool(self.conv2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool2")
             
             
         
        with tf.variable_scope('conv3_1'):
             W = tf.get_variable('weight', shape = [3,3,128,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool2,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv3_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv3_1/W",W)
             
        with tf.variable_scope('conv3_2'):
              W = tf.get_variable('weight', shape = [3,3,256,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv3_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv3_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv3_2/W",W)
              
        with tf.variable_scope('conv3_3'):
              W = tf.get_variable('weight', shape = [3,3,256,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv3_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv3_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv3_3/W",W)
             
        with tf.name_scope('pool3'):
             self.pool3= tf.nn.max_pool(self.conv3_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool3")
             
                                       
        with tf.variable_scope('conv4_1'):
             W = tf.get_variable('weight', shape = [3,3,256,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool3,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv4_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv4_1/W",W)
             
        with tf.variable_scope('conv4_2'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv4_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv4_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv4_2/W",W)
              
        with tf.variable_scope('conv4_3'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv4_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv4_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv4_3/W",W)
              
              ### keep it #####
              self.SSD_outputs.append(self.conv4_3)
              
             
        with tf.name_scope('pool4'):
             self.pool4= tf.nn.max_pool(self.conv4_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool4")
         
                          
             
        with tf.variable_scope('conv5_1'):
             W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool4,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv5_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv5_1/W",W)             

        with tf.variable_scope('conv5_2'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv5_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv5_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv5_2/W",W)
              
        with tf.variable_scope('conv5_3'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1), trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv5_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv5_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv5_3/W",W)
                                       
#        with tf.name_scope('pool5'):
#             self.pool5= tf.nn.max_pool(self.conv5_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool5")
#                   
#  #####################################################################################################################################################           
        
        with tf.variable_scope('SSD1'):
             
             W = tf.get_variable('weight', shape = [3, 3, 512, 1024], initializer = tf.truncated_normal_initializer(stddev= 0.01), trainable=True)
             b = tf.get_variable('bias', shape = [1024], initializer = tf.constant_initializer(0.0), trainable=True)
             conv = tf.nn.conv2d(self.conv5_3, W, strides = [1,1,1,1], padding = 'SAME')
             self.convSSD1 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv1/W",W)
             
        
        with tf.variable_scope('SSD2'):
             W = tf.get_variable('weight', shape = [1, 1, 1024, 1024], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [1024], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD1,W, strides = [1,1,1,1], padding = 'SAME')
             self.convSSD2 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv2/W",W)
             
             self.SSD_outputs.append(self.convSSD2)

   ###################################################################################################################################################             
             

        with tf.variable_scope('SSD3'):
             W = tf.get_variable('weight', shape = [1, 1, 1024, 256], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD2,W, strides = [1,1,1,1], padding = 'SAME')
             self.convSSD3 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)
             

        with tf.variable_scope('SSD4'):
             W = tf.get_variable('weight', shape = [3, 3, 256, 512], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD3,W, strides = [1,2,2,1], padding = 'SAME')
             self.convSSD4 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)
             
             self.SSD_outputs.append(self.convSSD4)

             
  ####################################################################################################################################################           

        with tf.variable_scope('SSD5'):
             W = tf.get_variable('weight', shape = [1, 1, 512, 128], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD4,W, strides = [1,1,1,1], padding = 'SAME')
             self.convSSD5 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)   
             
             
        with tf.variable_scope('SSD6'):
             W = tf.get_variable('weight', shape = [3, 3, 128, 256], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD5,W, strides = [1,2,2,1], padding = 'SAME')
             self.convSSD6 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)
             
             self.SSD_outputs.append(self.convSSD6)
             
 ###################################################################################################################################            
             
        with tf.variable_scope('SSD7'):
             W = tf.get_variable('weight', shape = [1, 1, 256, 128], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD6,W, strides = [1,1,1,1], padding = 'SAME')
             self.convSSD7 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)   
             
             
             
        with tf.variable_scope('SSD8'):
             W = tf.get_variable('weight', shape = [3, 3, 128, 256], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.convSSD7,W, strides = [1,2,2,1], padding = 'SAME')
             self.convSSD8 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             #self.poolF = tf.nn.avg_pool(self.convSSD8, [1, 3, 3, 1], [1, 1, 1, 1], padding='VALID')
             self.SSD_outputs.append(self.convSSD8)
            
             
 #########################################################################################################################################        
        
        
           
 





   
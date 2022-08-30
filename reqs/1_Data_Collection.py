#!/usr/bin/env python
# coding: utf-8

# In[90]:


import sys
import os
import glob
import numpy as np
from os import walk
import time
import regex as re
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import seaborn as sns
import sklearn 
import keras


# The "pipreqs" generates a puthon packages requirements text file based on the imports in the current jupyter notebook.
# 
# The required packages could be installed at once using pip and requirements.txt file

# In[99]:


get_ipython().system('pip install pipreqs')
get_ipython().system('pip install nbconvert')


# So what we’ve done here is converted our notebook into a .py file in a new directory called reqs, then run pipreqs in the new directory. The reason for this is that pipreqs only works on .py files and I can’t seem to get it to work when there are other files in the folder. The requirements.txt will be generated in the same folder.

# In[103]:


get_ipython().system('jupyter nbconvert --output-dir="./reqs" --to script 1_Data_Collection.ipynb')
get_ipython().system('cd reqs')
get_ipython().system('pipreqs --force')


# In[104]:


get_ipython().system('pip install -r requirements.txt')


# In[105]:


# Python version
print('Python: {}'.format(sys.version))
# pandas
print('pandas: {}'.format(pd.__version__))
# numpy
print('numpy: {}'.format(np.__version__))
# seaborn
print('seaborn: {}'.format(sns.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))
# Tensorflow-GPU
print('tensorflow: {}'.format(tf.__version__))


# ## Define functions - Read Images
# 
# The load_all_image_path function read all the files in the given directory and return a list of all file's path and the labels. Labels are merely the name of the image file on hard-drive. An example of files is
# 
# './Build2\\2020-03-08_13-14-42_layer_02955.jpg',
# 
# Whereas, a label is
# 
# '2020-03-08_13-14-42_layer_02955.jpg',

# In[20]:


def load_all_image_path(img_dir):
    
    #img_dir = "./Build2" # Enter Directory of all images
    img_labels = []
    for(_, _, filenames) in walk(img_dir):
        img_labels.extend(filenames)
        break
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    return files, img_labels


# ## Loading Images
# Read all the directorries in the given folder. It returns images paths and their labels.
# 
# __img_dir_paths__ = All Image's directory path/address.
# <br>
# __img_names__ = All Image's names.

# In[44]:


img_dir_paths, img_names = load_all_image_path("D:/UoH_PhD_Exp/Data/Build2")


# In[45]:


def var_info(var):
    print(type(var))
    print(len(var))
var_info(img_dir_paths)
var_info(img_names)


# Since out of all the images the first 1250 layers/images are relevant to our builts. That's why only the first 1250 are considered. For B1 and B2, the effective printing layers are 244-1242 and for B3 219-1217. But for simplicity, uniformity and avoiding complexity, first 1250 layers are selected.

# In[46]:


img_dir_paths = img_dir_paths[0:1250]
img_names = img_names[0:1250]
print(len(img_dir_paths))
print(len(img_names))


# ### Layers with porosity
# #### Old Labels

# In[30]:


b1_prosity_index = list(range(311,380)) + list(range(537, 554)) + list(range(628, 663)) + list(range(832, 862)) + list(range(936, 937)) + list(range(940, 953)) + list(range(1011, 1078)) + list(range(1145, 1152))
b2_prosity_index = list(range(311,380)) + list(range(428, 463)) + list(range(531, 560)) + list(range(640, 654)) + list(range(737, 753))
b3_prosity_index = list(range(420,456)) + list(range(519, 546)) + list(range(619, 634)) + list(range(719, 736)) + list(range(819, 827)) + list(range(919, 923))


# #### New Labels

# In[31]:


b1_remove_index = [311,312,313,318,320,325,326,335,340,366,369,374,375,376,537,538,539,540,541,542,543,544,545,546,547,
                       548,549,550,551,552,553,628,629,630,633,640,641,642,643,646,647,648,649,650,651,653,654,656,657,659,
                      661,662,833,832,833,834,835,836,837,838,840,842,843,844,845,846,847,849,850,851,852,853,855,857,936,
                       940,947,949,950,952,1011,1011,1012,1014,1018,1019,1020,1029,1030,1045,1075,1145,1146,1147,1148,1149,
                       1150,1151]


b2_remove_index = [320,324,429,430,431,432,433,434,437,450,451,452,456,459,462,531,532,533,534,535,536,537,
                      538,539,540,541,542,544,545,548,549,550,554,559,640,641,642,643,644,645,646,647,651,737,
                      740,741,742,743,744,745,748,750,751,752]


b3_remove_index = [420,423,425,436,439,442,449,453,519,521,522,533,534,538,541,542,543,620,621,622,627,629,
                      630,631,632,721,723,724,727,728,729,733,735,819,820,821,822,826,919,920,921,922]


# In[32]:


b1_prosity_index = [x for x in b1_prosity_index if x not in b1_remove_index]
print(b1_prosity_index[0:10])
print(len(b1_prosity_index))


# In[33]:


print(b1_prosity_index)


# In[34]:


b2_prosity_index = [x for x in b2_prosity_index if x not in b2_remove_index]
print(b2_prosity_index[0:10])
print(len(b2_prosity_index))


# In[35]:


print(b2_prosity_index)


# In[36]:


b3_prosity_index = [x for x in b3_prosity_index if x not in b3_remove_index]
print(b3_prosity_index[0:10])
print(len(b3_prosity_index))


# In[37]:


print(b3_prosity_index)


# ### Image selection and Cropping

# > The following function receives a chunk of image's path and their corresponding labels. Not all the images in Build2 are relevant to our cylinders. Out of total 2922 images, only 963 images relevant to our 3d objects. Three cylinders names as B1, B2, B3 were printed. Images from 243 to 1243 are related to B1 and B2 cylinders. Whereas, B3 cylinder related images are ranges from 218 to 1218. 
# <br><br><br>
# Firstly, the images were read into a numpy array called imgs_data. The image dimensions are __height = 2600 and Width = 1420__. Each image is then cropped into three small sections. __Height=1250-1440 and width=650-1100__ is firstly croped from the whole powder bed image.  <br> <br>
# The cropped image is further is divided into three parts, each containg the image of a cylinder[B1,B2,B3]. The coordinates of __B1=[h:0-190, w:0-150]__, __B2 = [h:0-190, w:150-300]__ , __B3 = [h:0-190, w:300-450]__. The three images were then stored in different folders on the hard-drive.  

# In[38]:


def crop_save_images(files, directory, labels):
   
    for f1,lab in zip(files,labels):
        #F1 = File path.
        #lab = Image label
        #print("F1: " + str(f1))
        #print("Lab: " +  str(lab))
        ########## read image
        orig_img = cv2.imread(f1)

        ########### crop image
        img = orig_img[1250:1440, 650:1100]
        img1 = img[0:190,0:150]
        img2 = img[0:190,150:300]
        img3 = img[0:190,300:450]

        ########### Label Image
        
        tt = lab[:-4].split('_')
        #tt = layer number
        #print(tt[3])
        layer_no = int(tt[3])
        
        if (layer_no in b1_prosity_index):
            img_name_b1 = "1_B1_Layer_"+str(layer_no)+".jpg"
            #print(layer_no)
            #print("True")
        else:
            img_name_b1 = "0_B1_Layer_"+str(layer_no)+".jpg"
            #print("False")
            
        if (layer_no in b2_prosity_index):
            img_name_b2 = "1_B2_Layer_"+str(layer_no)+".jpg"
            #print(layer_no)
            #print("True")
        else:
            img_name_b2 = "0_B2_Layer_"+str(layer_no)+".jpg"
            #print("False")
            
        if (layer_no in b3_prosity_index):
            img_name_b3 = "1_B3_Layer_"+str(layer_no)+".jpg"
            #print(layer_no)
            #print("True")
        else:
            img_name_b3 = "0_B3_Layer_"+str(layer_no)+".jpg"
            #print("False")
        ########### store image
        if(layer_no>243 and layer_no<1243):
            img_name = directory[0] + img_name_b1
            matplotlib.image.imsave(img_name, img1)
        
            img_name = directory[1] + img_name_b2
            matplotlib.image.imsave(img_name, img2)
        if(layer_no>218 and layer_no<1218):
            img_name = directory[2] + img_name_b3
            matplotlib.image.imsave(img_name, img3)
        #break


# In[72]:


def crop_save_images_with_poreSize(files, directory, labels):
    
    for f1,lab in zip(files,labels):
        #F1 = File path.
        #lab = Image label
        #print("F1: " + str(f1))
        #print("Lab: " +  str(lab))
        ########## read image
        orig_img = cv2.imread(f1)

        ########### crop image
        img = orig_img[1250:1440, 650:1100]
        img1 = img[0:190,0:150]
        img2 = img[0:190,150:300]
        img3 = img[0:190,300:450]

        ########### Label Image
        
        tt = lab[:-4].split('_')
        #tt = layer number
        #print(tt[3])
        layer_no = int(tt[3])
        #########################################################################################-----B1
        if (layer_no in b1_prosity_index):
            ############################################################### 2mm
            if(layer_no >= 314 and layer_no <=379) or (layer_no >= 1013 and layer_no <=1077):
                img_name_b1 = "1_B1_Layer_"+str(layer_no)+"_2mm"+".jpg"
            ############################################################### 1mm
            elif (layer_no >= 631 and layer_no <=660) or (layer_no >= 839 and layer_no <=861):
                img_name_b1 = "1_B1_Layer_"+str(layer_no)+"_1mm"+".jpg"
            ############################################################### 0.5mm
            elif (layer_no >= 941 and layer_no <=951):
                img_name_b1 = "1_B1_Layer_"+str(layer_no)+"_05mm"+".jpg"
        else:
            img_name_b1 = "0_B1_Layer_"+str(layer_no)+".jpg"
            #print("False")
        #########################################################################################-----B2
        if (layer_no in b2_prosity_index):
            ############################################################### 2mm
            if(layer_no >= 311 and layer_no <=379):
                img_name_b2 = "1_B2_Layer_"+str(layer_no)+"_2mm"+".jpg"
            ############################################################### 1mm
            elif (layer_no >= 428 and layer_no <=461):
                img_name_b2 = "1_B2_Layer_"+str(layer_no)+"_1mm"+".jpg"
            ############################################################### 0.8mm
            elif (layer_no >= 543 and layer_no <=558):
                img_name_b2 = "1_B2_Layer_"+str(layer_no)+"_08mm"+".jpg"
            ############################################################### 0.5mm
            elif (layer_no >= 648 and layer_no <=653):
                img_name_b2 = "1_B2_Layer_"+str(layer_no)+"_05mm"+".jpg"
            ############################################################### 0.4mm
            elif (layer_no >= 738 and layer_no <=749):
                img_name_b2 = "1_B2_Layer_"+str(layer_no)+"_04mm"+".jpg"
            
        else:
            img_name_b2 = "0_B2_Layer_"+str(layer_no)+".jpg"
            #print("False")
        #########################################################################################-----B3
        if (layer_no in b3_prosity_index):
            ############################################################### 1mm
            if(layer_no >= 421 and layer_no <=455):
                img_name_b3 = "1_B3_Layer_"+str(layer_no)+"_1mm"+".jpg"
            ############################################################### 0.8mm
            elif (layer_no >= 520 and layer_no <=545):
                img_name_b3 = "1_B3_Layer_"+str(layer_no)+"_08mm"+".jpg"
            ############################################################### 0.5mm
            elif (layer_no >= 619 and layer_no <=633):
                img_name_b3 = "1_B3_Layer_"+str(layer_no)+"_05mm"+".jpg"
            ############################################################### 0.4mm
            elif (layer_no >= 719 and layer_no <=734):
                img_name_b3 = "1_B3_Layer_"+str(layer_no)+"_04mm"+".jpg"
            ############################################################### 0.2mm
            elif (layer_no >= 823 and layer_no <=825):
                img_name_b3 = "1_B3_Layer_"+str(layer_no)+"_02mm"+".jpg"
            #print(layer_no)
            #print("True")
        else:
            img_name_b3 = "0_B3_Layer_"+str(layer_no)+".jpg"
            #print("False")
        ########### store image
        if(layer_no>243 and layer_no<1243):
            img_name = directory[0] + img_name_b1
            matplotlib.image.imsave(img_name, img1)
        
            img_name = directory[1] + img_name_b2
            matplotlib.image.imsave(img_name, img2)
        if(layer_no>218 and layer_no<1218):
            img_name = directory[2] + img_name_b3
            matplotlib.image.imsave(img_name, img3)
        #break


# Since out of all the images the first 1250 layers/images are relevant to our builts. That's why only the first 1250 are considered. For B1 and B2, the effective printing layers are 244-1242 and for B3 219-1217. But for simplicity, uniformity and avoiding complexity, first 1250 layers are selected.

# In[49]:


directories = ["D:/UoH_PhD_Exp/Data/Crop_images/B1/", "D:/UoH_PhD_Exp/Data/Crop_images/B2/", "D:/UoH_PhD_Exp/Data/Crop_images/B3/"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        


# #### Remove all the old files in B1, B2 & B3 folder

# In[56]:


for directory in directories:
    files = glob.glob(os.path.join(directory,"*"))
    for f in files:
        os.remove(f)


# #### Cropping images

# In[57]:


crop_save_images(img_dir_paths[217:1206] ,directories, img_names[217:1206]) 


# ##  Loading cropped images
# ### B1 Images

# In[64]:


files, labels = load_all_image_path("D:/UoH_PhD_Exp/Data/Crop_images/B1/")
#print(labels[0])
#print(files[0])
data = []
b1_labels = list()
b1_layer_numbers = list()
for f1, lab in zip(files, labels):
    #print("lab:" + lab)
    layer_num = re.search('Layer_(.+?).jpg', lab).group(1)
    b1_layer_numbers.append("b1_"+str(layer_num))
    b1_labels.append(int(lab[0]))
    img = cv2.imread(f1)
    ######### Convert to Images to grey scale.
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(img)
b1_images = np.array(data)
print(b1_images[0].shape)
print(b1_labels[0])
print(b1_layer_numbers[0])
(unique, counts) = np.unique(b1_labels, return_counts=True)
print(unique, counts)
print(b1_images.shape)


# In[66]:


files, labels = load_all_image_path("D:/UoH_PhD_Exp/Data/Crop_images/B2/")
data = []
b2_labels = list()
b2_layer_numbers = list()
for f1, lab in zip(files, labels):
    layer_num = re.search('Layer_(.+?).jpg', lab).group(1)
    b2_layer_numbers.append("b2_"+str(layer_num))
    img = cv2.imread(f1)
    b2_labels.append(int(lab[0]))
    ######## Convert to Images to grey scale.
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(img)
b2_images = np.array(data)
print(b2_images[0].shape)
print(b2_labels[0])
print(b2_layer_numbers[0])
(unique, counts) = np.unique(b2_labels, return_counts=True)
print(unique, counts)
print(b2_images.shape)


# In[67]:


files, labels = load_all_image_path("D:/UoH_PhD_Exp/Data/Crop_images/B3")
data = []
b3_labels = list()
b3_layer_numbers = list()
for f1, lab in zip(files, labels):
    layer_num = re.search('Layer_(.+?).jpg', lab).group(1)
    b3_layer_numbers.append("b3_"+str(layer_num))
    img = cv2.imread(f1)
    b3_labels.append(int(lab[0]))
    ######## Convert to Images to grey scale.
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(img)
b3_images = np.array(data)
print(b3_images[0].shape)
print(b3_labels[0])
print(b3_layer_numbers[0])
(unique, counts) = np.unique(b3_labels, return_counts=True)
print(unique, counts)
print(b3_images.shape)


# #### Concatenate all images datasets into one dataset
# 
# X = Images     y = Image labels

# In[71]:


X = np.concatenate((b1_images, b2_images, b3_images), axis=0)
y = b1_labels + b2_labels + b3_labels
layer_nums = b1_layer_numbers + b2_layer_numbers + b3_layer_numbers

print("X Shape: " + str(X.shape))
print("Total y: " + str(len(y)))


# In[ ]:





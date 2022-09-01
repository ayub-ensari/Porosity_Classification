#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from os import walk
import time
import regex as re
import os
import glob
import tensorflow as tf
import warnings


# In[ ]:


from platform import python_version
print("Python: " + python_version())


# In[ ]:


#!pipreqs --print


# In[ ]:


print("Checking the avaiable GPU devices. \n")
if not tf.test.gpu_device_name():
    warnings.warn("No GPU found")
else:
    print("Default GPU device: {}".format(tf.test.gpu_device_name()))


# In[ ]:


def load_all_image_path(img_dir):
    
    #img_dir = "./Build2" # Enter Directory of all images
    img_labels = []
    for(_, _, filenames) in walk(img_dir):
        img_labels.extend(filenames)
        break
    data_path = os.path.join(img_dir,'*g')
    files = glob.glob(data_path)
    return files, img_labels
    


# In[ ]:


img_paths, labels = load_all_image_path("D:/UoH_PhD_Exp/Data//Build2")
print(len(img_paths))
print(len(labels))


# ### Layers with porosity

# In[ ]:


b1_prosity_index = list(range(311,380)) + list(range(537, 554)) + list(range(628, 663)) + list(range(832, 862)) + list(range(936, 937)) + list(range(940, 953)) + list(range(1011, 1078)) + list(range(1145, 1152))
b2_prosity_index = list(range(311,380)) + list(range(428, 463)) + list(range(531, 560)) + list(range(640, 654)) + list(range(737, 753))
b3_prosity_index = list(range(420,456)) + list(range(519, 546)) + list(range(619, 634)) + list(range(719, 736)) + list(range(819, 827)) + list(range(919, 923))


# The following function receives a chunk of image's path and their corresponding labels. Firstly, the images were read into a numpy array called imgs_data. Each image is then cropped into three small sections. Height=1250-1440 and width=650-1100 is firstly croped from the whole powder bed image. The cropped image is further is divided into three parts, each containg the image of a cylinder[B1,B2,B3]. The coordinates of B1=[h:0-190, w:0-150], B2 = [h:0-190, w:150-300] , B3 = [h:0-190, w:300-450]. The three images were then stored in different folders on the hard-drive.  

# In[ ]:


def crop_save_images(files, directory, labels):
    b1_prosity_index = list(range(311,380)) + list(range(537, 554)) + list(range(628, 663)) + list(range(832, 862)) + list(range(936, 937)) + list(range(940, 953)) + list(range(1011, 1078)) + list(range(1145, 1152))
    b2_prosity_index = list(range(311,380)) + list(range(428, 463)) + list(range(531, 560)) + list(range(640, 654)) + list(range(737, 753))
    b3_prosity_index = list(range(420,456)) + list(range(519, 546)) + list(range(619, 634)) + list(range(719, 736)) + list(range(819, 827)) + list(range(919, 923))
    
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


# Since out of all the images the first 1250 layers/images are relevant to our builts. That's why only the first 1250 are considered. For B1 and B2, the effective printing layers are 244-1242 and for B3 219-1217. But for simplicity, uniformity and avoiding complexity, first 1250 layers are selected.

# In[ ]:


directories = ["./Crop_images/B1/", "./Crop_images/B2/", "./Crop_images/B3/"]
crop_save_images(img_paths[217:1206] ,directories, labels[217:1206]) 


# ##  Loading cropped images

# In[ ]:


files, labels = load_all_image_path("./Crop_images/B1")
data = []
b1_labels = list()
b1_layer_numbers = list()
for f1, lab in zip(files, labels):
    layer_num = re.search('Layer_(.+?).jpg', lab).group(1)
    b1_layer_numbers.append("b1_"+str(layer_num))
    b1_labels.append(int(lab[0]))
    img = cv2.imread(f1)
    ######### Convert to Images to grey scale.
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data.append(img)
b1_images = np.array(data)


# In[ ]:


(unique, counts) = np.unique(b1_labels, return_counts=True)
print(unique, counts)
print(b1_images.shape)


# In[ ]:


files, labels = load_all_image_path("./Crop_images/B2")
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


# In[ ]:


(unique, counts) = np.unique(b2_labels, return_counts=True)
print(unique, counts)
print(b2_images.shape)


# In[ ]:


files, labels = load_all_image_path("./Crop_images/B3")
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


# In[ ]:


(unique, counts) = np.unique(b3_labels, return_counts=True)
print(unique, counts)
print(b3_images.shape)


# #### Concatenate all images datasets into one

# X = images.      y = image Labells

# In[ ]:


X = np.concatenate((b1_images, b2_images, b3_images), axis=0)
y = b1_labels + b2_labels + b3_labels
layer_nums = b1_layer_numbers + b2_layer_numbers + b3_layer_numbers


# In[ ]:


print(X.shape)
print(len(y))


# ## Working with HDF5 dataset formate
# 
# #### Source: https://realpython.com/storing-images-in-python/

# ### Store all images as hdf5 formate on hard drive

# In[ ]:


def store_many_hdf5(images, labels, file_path):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    hdf5_dir = file_path
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()


# In[ ]:


dir_file_path = "./HDF5_Dataset/Build2.h5"
store_many_hdf5(X,y, dir_file_path)


# ### Reading Images as HDf5

# In[ ]:


def read_many_hdf5(num_images, file_path):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(file_path, "r+")

    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")

    return images, labels


# In[ ]:


dir_file_path = "./HDF5_Dataset/Build2.h5"
XX, yy = read_many_hdf5(0,dir_file_path)


# In[ ]:


print(XX.shape)
print(len(yy))


# In[ ]:





# # Preprocessing Images
# 
# ## 1. Image Channels + Brightness

# In[ ]:


def visualize_n_imgs(test_images,test_titles, n):
    plt.figure(figsize=(20,10))
    for i in range(n):
        plt.subplot(1,n,i+1),plt.imshow(test_images[i]), plt.title(test_titles[i]), plt.xticks([]),plt.yticks([])

    plt.show()


# In[ ]:


def simple_images(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]#:,:,:]
        rnd_images.append(img)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def lab_grey_images(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        im1 = rgb2lab(img)
        im1[...,1] = im1[...,2] = 0
        im1 = lab2rgb(im1)
        
        rnd_images.append(im1)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def grey_images(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        im1 = rgb2gray(img)
        
        rnd_images.append(im1)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def lab_bright_images(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        im1 = rgb2lab(img)
        im1[...,0] = im1[...,0] + 30
        im1 = lab2rgb(im1)
        
        rnd_images.append(im1)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# ## 2. Edge Detection

# In[ ]:


from medpy.filter.smoothing import anisotropic_diffusion
from skimage.filters import gaussian, threshold_otsu
from skimage import util


# In[ ]:


def normalize(img):
    return (img-np.min(img))/(np.max(img)-np.min(img))

def sketch(img, edges):
    output = np.multiply(img, edges)
    output[output>1]=1
    output[edges==1]=1
    #output = normalize(output)
    return output

def edges_with_anisotropic_diffusion(img, niter=100, kappa=10, gamma=0.1):
    #img = gaussian(img, sigma=0.05)
    output = img - anisotropic_diffusion(img, niter=niter, kappa=kappa, gamma=gamma, voxelspacing=None, option=1)
    output[output > 0] = 1
    output[output < 0] = 0
    #output = np.clip(output, 0, 1)
    #thresh = threshold_otsu(output)
    #output = np.invert(output > thresh)
    return output

def edges_with_dodge2(img):
    img_blurred = gaussian(util.invert(img), sigma=5)
    output = np.divide(img, util.invert(img_blurred) + 0.001) # avoid division by zero
    print(np.max(output), np.min(output))
    output = normalize(output)
    thresh = threshold_otsu(output)
    output = output > thresh
    return output


def sketch_with_dodge(img):
    orig = img
    blur = gaussian(util.invert(img), sigma=20)
    result=blur/util.invert(orig) 
    result[result>1]=1
    result[orig==1]=1
    return result

# with DOG
def edges_with_DOG(img, k = 200, gamma = 1):
    sigma = 0.5
    output = gaussian(img, sigma=sigma) - gamma*gaussian(img, sigma=k*sigma)
    output[output > 0] = 1
    output[output < 0] = 0 
    return output

def sketch_with_XDOG(image, epsilon=0.01):
  """
  Computes the eXtended Difference of Gaussians (XDoG) for a given image. This 
  is done by taking the regular Difference of Gaussians, thresholding it
  at some value, and applying the hypertangent function the the unthresholded
  values.
  image: an n x m single channel matrix.
  epsilon: the offset value when computing the hypertangent.
  returns: an n x m single channel matrix representing the XDoG.
  """
  phi = 10

  difference = edges_with_DOG(image, 200, 0.98).astype(np.uint8)
  #difference = sketch(image, difference)
  #difference = normalize(difference)  

  for i in range(0, len(difference)):
    for j in range(0, len(difference[0])):
      if difference[i][j] >= epsilon:
        difference[i][j] = 1
      else:
        ht = np.tanh(phi*(difference[i][j] - epsilon))
        difference[i][j] = 1 + ht
  difference = normalize(difference)  
  return difference


# In[ ]:


def aniso(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        output_aniso = sketch(img, edges_with_anisotropic_diffusion(img))
        rnd_images.append(output_aniso)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def dog(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        output_dog = sketch(img, edges_with_DOG(img, k=25))
        rnd_images.append(output_dog)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def xdog(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        output_xdog = sketch_with_XDOG(img)
        rnd_images.append(output_xdog)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def dodge(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        output_dodge = sketch_with_dodge(img)
        rnd_images.append(output_dodge)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def dodge2(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        output_dodge2 = sketch(img, edges_with_dodge2(img))
        rnd_images.append(output_dodge2)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# ## Thresholding
# ### 1. Binary Thresholding

# In[ ]:


def inverse_binary_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        #img = rgb2gray(img)
        ret,thresh2 = cv2.threshold(img,60,255,cv2.THRESH_BINARY_INV)
        rnd_images.append(thresh2)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def Trunc_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        ret,thresh3 = cv2.threshold(img,60,255,cv2.THRESH_TRUNC)
        rnd_images.append(thresh3)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def tozero_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        #img = rgb2gray(img)
        ret,thresh4 = cv2.threshold(img,60,255,cv2.THRESH_TOZERO)
        rnd_images.append(thresh4)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


def tozero_inverse_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        img = rgb2gray(img)
        ret,thresh5 = cv2.threshold(img,60,255,cv2.THRESH_TOZERO_INV)
        rnd_images.append(thresh5)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:





# In[ ]:





# In[ ]:





# Get 12 random images from the dataset

# In[ ]:


n=12
rnd_indx = np.random.choice(XX.shape[0], n, replace=False)
print("Randome selected image's indexes")
rnd_indx[0] = 1060
print(rnd_indx)


# In[ ]:


simple_images(XX,yy,rnd_indx,n)


# In[ ]:


lab_grey_images(XX,yy,rnd_indx,n)


# In[ ]:


grey_images(XX,yy,rnd_indx,n)


# In[ ]:


lab_bright_images(XX,yy,rnd_indx,n)


# In[ ]:


aniso(XX,yy, rnd_indx, n)


# In[ ]:


dog(XX,yy, rnd_indx, n)


# In[ ]:


xdog(XX,yy, rnd_indx, n)


# In[ ]:


dodge(XX,yy, rnd_indx, n)


# In[ ]:


dodge2(XX,yy, rnd_indx, n)


# In[ ]:


def binary_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        #img = rgb2gray(img)
        ret,thresh1 = cv2.threshold(img,65,255,cv2.THRESH_BINARY)
        rnd_images.append(thresh1)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)


# In[ ]:


imgg = XX[rnd_indx[0]]
print(imgg.shape)


# In[ ]:


simple_images(XX,yy,rnd_indx,n)


# In[ ]:


binary_thd(XX,yy, rnd_indx, n)


# In[ ]:


inverse_binary_thd(XX,yy, rnd_indx, n)


# In[ ]:


Trunc_thd(XX,yy, rnd_indx, n)


# In[ ]:


tozero_thd(XX,yy, rnd_indx, n)


# In[ ]:


tozero_inverse_thd(XX,yy, rnd_indx, n)


# ### Ostu Thresholding

# In[ ]:


def ostu_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        #img = rgb2gray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        rnd_images.append(th2)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)
    
def ostu_gussian_thd(XX,yy, rnd_indx, n):
    rnd_images = []
    test_titles = []
    for i in rnd_indx:
        test_titles.append(yy[i])
        img = XX[i]
        #img = rgb2gray(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        rnd_images.append(th3)
    test_images = np.array(rnd_images)
    visualize_n_imgs(test_images,test_titles, n)   


# In[ ]:


ostu_thd(XX,yy, rnd_indx, n)


# In[ ]:


ostu_gussian_thd(XX,yy, rnd_indx, n)


# In[ ]:





# ### Data Pre processing

# ### Reshaping Data to adjust greyscal dimension

# In[ ]:


# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 190, 150, 1))
X_test = X_test.reshape((X_test.shape[0], 190, 150, 1))


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


# Convert the array to float32 as opposed to uint8
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Convert the pixel values from integers between 0 and 255 to floats between 0 and 1
X_train /= 255
X_test /=  255


# In[ ]:


NUM_DIGITS = 2

print("Before", y[1210]) # The format of the labels before conversion

y_train  = tf.keras.utils.to_categorical(y_train, NUM_DIGITS)

print("After", y[1210]) # The format of the labels after conversion

y_test = tf.keras.utils.to_categorical(y_test, NUM_DIGITS)


# In[ ]:


print("Train y: " + str(y_train[0:5]))
print("Test y: " + str(y_test[0:5]))


# ### GPU Configuration

# In[ ]:


print("Num Devices Available: ", len(tf.config.experimental.list_physical_devices()))


# In[ ]:


print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))


# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


#import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)


# In[ ]:


#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
  # Restrict TensorFlow to only use the first GPU
#  try:
#    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
#    print(e)


# In[ ]:


#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#from tensorflow.keras import backend
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#backend.set_session(sess)


# ### Model

# In[ ]:


np.random.seed(786)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(190, 150,1)))
model.add(tf.keras.layers.MaxPool2D(strides=2))
model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(strides=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(28500,)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


# In[ ]:


print(X_train.shape)
print(np.unique(y_train, return_counts=True))


# ### Class Weight  

# In[ ]:


weight_for_0 = 0.3
weight_for_1 = 0.7
my_class_weight = {0: weight_for_0, 1: weight_for_1}


# In[ ]:


#model.fit(samples_cnn, dataset.labels, epochs=epochs, batch_size=batch_size, verbose=1)
history = model.fit(X_train, y_train, validation_split=0.33, epochs=50,verbose=1, class_weight=my_class_weight)#, batch_size=250, verbose=1)


# ## Save Model weights

# In[ ]:


# always save your weights after training or during training
model.save_weights('new_model.h5')  


# In[ ]:


print(history)


# In[ ]:


loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: %.2f' % (accuracy))


# In[ ]:


y_pred = model.predict(X_test)


# #### Convert class attribute back to its origional form, 0,1

# In[ ]:


y_actual = np.argmax(y_test,axis=1)
print(y_actual[0:25])
y_pred = np.argmax(y_pred,axis=1)
print(y_pred[0:25])


# #### Classification Report

# In[ ]:


#y_actual = np.argmax(y_test,axis=1)
print(y_actual[0:5])
#y_pred = np.argmax(y_pred,axis=1)
#Accuracy of the predicted values
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
print(classification_report(y_actual,y_pred))
print(confusion_matrix(y_actual,y_pred))
print(accuracy_score(y_actual,y_pred))


# #### Find out the images wrongly classified. we will find out their indexes by comparing actual and predicte labels.
# 
# The indexs will be used to print the images in the later stage.

# In[ ]:


j = 0
# Output list intialisation 
wrongClassiified_indexes = []    
# Using iteration to find 
for i in y_actual: 
    if i != y_pred[j]: 
        wrongClassiified_indexes.append(j) 
    j = j + 1


# In[ ]:


print(wrongClassiified_indexes)
print("Total Wrong Imgaes: " + str(len(wrongClassiified_indexes)))


# In[ ]:


for i in range(0, len(wrongClassiified_indexes)):
    index = wrongClassiified_indexes[i]
    print("Prd: " + str(y_pred[index]) + "Act:" + str(y_actual[index]) + "Layer: " + str(layer_nums[y_indeces[index]]))


# ### Rescalling back to three dimentions for visualization

# In[ ]:


# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 190, 150))
X_test = X_test.reshape((X_test.shape[0], 190, 150))


# In[ ]:


n = 16
data = []
titles = []
for i in range(0,n):
    index = wrongClassiified_indexes[i]
    titles.append("Prd: " + str(y_pred[index]) + "Act:" + str(y_actual[index]) + "Layer: " + str(layer_nums[y_indeces[index]]))
    img = X_test[index]#:,:,:]
    data.append(img)
test_images = np.array(data)

plt.figure(figsize=(20,10))
for i in range(n):
    plt.subplot(n/4,4,i+1),plt.imshow(test_images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()


# In[ ]:


plt.imshow(X_test[4])
plt.show()
print("Predict Label:" + str(y_pred[4]))
print("Actual Label:" + str(y_actual[4]))


# In[ ]:


############### printing accuracy and loss between the epoches #########
import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'], label="Training")
plt.plot(history.history['val_accuracy'], label ="Validation")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
#plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'], label="Training")
plt.plot(history.history['val_loss'], label ="Validation")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# ## Improvement Point:
# > - I think this happens when the validation set does not reflect the reality of the test set. So using k-fold technique my be helpful in making the validation result similar to testing result.
# > - Use Data Augmentation techniques for class imbalance.

# In[ ]:





# In[ ]:


#%reset


# In[ ]:





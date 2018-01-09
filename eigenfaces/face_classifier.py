#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:27:23 2018

@author: dlw
"""

import logging
import numpy as np 
import os
import re
import imageio
import random
from matplotlib import pyplot as plt
from PIL import Image

IMG_SIZE = 64

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
logging.debug('Start of program')

class face_classifier():
    
    def __init__(self):
        self.learningset.nonsmiling   = []
        self.learningset.smiling      = []
        self.learningset.mean_s       = []
        self.learningset.mean_ns      = []
        self.testingset.nonsmiling    = []  
        self.testingset.smiling       = []
        self.eigenfaces               = []
    
    
    


def import_images(dir='.'):
    #imports images from directory DIR (defaults to current directory)
    images = []
    path, dirs, files = next(os.walk(dir))
    logging.debug('Importing ' + str(len(files))+ ' images from ' + dir)
    for filename in files:
        images.append(imageio.imread(dir + filename)) 
    return images;

def mean_image(images):
    #computes mean image of set IMAGES
    tot_image = np.empty(images[0].shape)
    for image in images:
        tot_image = tot_image + image
    m_image = tot_image/len(images)
    #scale to 255
    m_image = np.uint8(m_image*255/m_image.max())
    
    return m_image;

def resize_images(images):
    new_images = []
    for image in images:
        new_images.append(resize(image))
    return new_images;

def pad(image):
    pad_size = abs(max(image.shape)-min(image.shape))
    pad_image = np.pad(image, ((pad_size//2,pad_size//2),(0,0)), 'constant', constant_values=(0, 0))
    return pad_image;

def pad_images(images):
    padded_images = []
    for image in images:
        padded_images.append(pad(image))
    return padded_images;

def resize(image, size=64):
    #calculate the resize factor 
    rs_factor = size/max(image.shape)
    new_size  = tuple([int(round(num*rs_factor)) for num in image.shape])
    #use Image.resize function to convert 
    im = Image.fromarray(image)
    im = im.resize(new_size, Image.ANTIALIAS)
    #convert back to np array
    new_image = np.array(im)
    return new_image;
  
def sub_mean_image(images, m_image):
    new_images = []
    for image in images:
        new_images.append(image - m_image)
    
    return new_images;

def image2vector(image):
    v_image = np.reshape(image, tuple([1,np.prod(image.shape)]))
    return v_image;

def vector2image(v_image, shape):
    image = np.reshape(v_image, shape)
    #scale to 255 uint
    #image = np.uint8(image*255/image.max())
    return image;

def vectorize_images(images):
    v_images = []
    for image in images:
        v_images.append(image2vector(image))
    return v_images;

def cat_vectors(images_v):
    image_samples = len(images_v)
    image_len     = images_v[0].shape[1]
    images_a = np.empty((image_samples, image_len))
    for k,image_v in enumerate(images_v):
        images_a[k,:] = image_v
    
    return images_a.astype('float')

def normalize(image):
    n_image = (255*(image - np.max(image))/-np.ptp(image)).astype(int)
    
    return n_image;

#import image into workspace
face_dir = '../images/faces/'
smiling_dir = '../images/smiling_faces/'

#image file format - 1a.jpg
re_face_key = re.compile(r'\d+[a].\w{3}')

#import non-smiling images
images_ns = import_images(face_dir)
#import smiling images
images_s = import_images(smiling_dir)

#compute mean face
logging.debug('Computing mean images')
m_image_ns = mean_image(images_ns)
m_image_s  = mean_image(images_s)

#create training and testing set
logging.debug('Generating Training and Testing Sets')
random.shuffle(images_s)
random.shuffle(images_ns)
training_images_s  = images_s[0:len(images_s)//2]
training_images_ns = images_ns[0:len(images_ns)//2]
testing_images_s  = images_s[len(images_s)//2:]
testing_images_ns = images_ns[len(images_ns)//2:]

#training
#downsize to 64x64 with zero pad
logging.debug('Resizing Training Set')
training_images_s = pad_images(resize_images(training_images_s))
training_images_ns = pad_images(resize_images(training_images_ns))
m_image_s = pad(resize(m_image_s))
m_image_ns = pad(resize(m_image_ns))
    
if False:
    plt.figure()
    plt.imshow(m_image_s, cmap='gray')
    plt.figure()
    plt.imshow(m_image_ns, cmap='gray')
    plt.figure()
    plt.imshow(training_images_s[0])

training_images_v_s = vectorize_images(training_images_s)
training_images_v_ns = vectorize_images(training_images_ns)

#Subtract mean from images
training_images_v_s = sub_mean_image(training_images_v_s, image2vector(m_image_s))
training_images_v_ns = sub_mean_image(training_images_v_ns, image2vector(m_image_ns))

#calculate covariance of the image set
logging.debug('Calculating Covariance of Image Set')
training_array_s = cat_vectors(training_images_v_s)
image_cov_s = np.cov(np.transpose(training_array_s))
training_array_ns = cat_vectors(training_images_v_ns)
image_cov_ns = np.cov(np.transpose(training_array_ns))
#%%

logging.debug('Performing SVD on Image Covariance Matrix')
U_s, E_s, Vh_s = np.linalg.svd(image_cov_s, full_matrices=False)
U_ns, E_ns, Vh_ns = np.linalg.svd(image_cov_ns, full_matrices=False)
#%%

eigen_faces_s = []
eigen_faces_ns = []
for k in range(16):
    eigen_face_v_s = normalize(U_s[:,k])
    eigen_faces_s.append(vector2image(eigen_face_v_s, tuple([IMG_SIZE, IMG_SIZE])))
    eigen_face_v_ns = normalize(U_ns[:,k])
    eigen_faces_ns.append(vector2image(eigen_face_v_ns, tuple([IMG_SIZE, IMG_SIZE])))
    
    
    
#plt.figure()
#plt.imshow(eigen_faces_s[0], cmap='gray')
#plt.figure()
#plt.imshow(eigen_faces_s[1], cmap='gray')
#plt.figure()
#plt.imshow(eigen_faces_s[2], cmap='gray')
#plt.figure()
#plt.imshow(eigen_faces_s[3], cmap='gray')

#show first 16 eigenfaces
fig_s = plt.figure()

for k in range(16):
    plt.subplot(4,4,k+1)
    plt.axis('off')
    plt.imshow(eigen_faces_s[k], cmap='gray')

fig_ns = plt.figure()
    
for k in range(16):
    plt.subplot(4,4,k+1)
    plt.axis('off')
    plt.imshow(eigen_faces_ns[k], cmap='gray')
    

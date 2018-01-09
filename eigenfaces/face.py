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

class face_classifier(object):
    
    def __init__(self):
        logging.debug('__init__ class')
        self.learning_set_a = None
        self.learning_set_b = None
        self.mean_a         = None
        self.mean_b         = None
        self.eigenfaces_a   = None
        self.eigenfaces_b   = None
        self.image_size     = 64
    
    def get_learning_set_a(self):
        return self.learning_set_a
    
    def get_learning_set_b(self):
        return self.learning_set_b
    
    def get_mean_a(self):
        return self.mean_a
    
    def get_mean_b(self):
        return self.mean_b
    
    def get_eigenfaces_a(self):
        return self.eigenfaces_a
    
    def get_eigenfaces_b(self):
        return self.eigenfaces_b
    
    def calc_eigenfaces(self, images, m_image):
        images_v = self.vectorize_images(images)
        images_v_m = self.sub_mean_image(images_v, self.image2vector(m_image))
        training_array = self.cat_vectors(images_v_m)
        image_cov = np.cov(np.transpose(training_array))
        #logging.debug('Performing SVD on Image Covariance Matrix')
        U, E, Vh = np.linalg.svd(image_cov, full_matrices=False)
        eigenface_set = []
        for k in range(16):
            eigenface_v = self.normalize(U[:,k])
            eigenface_set.append(self.vector2image(eigenface_v, tuple([self.image_size, self.image_size])))

        return eigenface_set;
    
    def set_learning_set_a(self, dir):
        logging.debug('calling self.get_learning_set_a')
        self.learning_set_a = self.set_learning_set(dir)
        self.mean_a         = self.mean_image(self.learning_set_a)
        self.eigenfaces_a   = self.calc_eigenfaces(self.learning_set_a, self.mean_a)
     
    def set_learning_set_b(self, dir):
        logging.debug('calling self.get_learning_set_b')
        self.learning_set_b = self.set_learning_set(dir)
        self.mean_b         = self.mean_image(self.learning_set_b)
        self.eigenfaces_b   = self.calc_eigenfaces(self.learning_set_b, self.mean_b)
    
    def set_learning_set(self, dir):
        logging.debug('calling get_learning_set')
        return self.pad_images(self.resize_images(self.import_images(dir)))

    def import_images(self, dir):
        #imports images from directory DIR (defaults to current directory)
        logging.debug('calling import images')
        images = []
        path, dirs, files = next(os.walk(dir))
        logging.debug('Importing ' + str(len(files))+ ' images from ' + dir)
        for filename in files:
            if filename[-3:]=='jpg':
                images.append(imageio.imread(dir + filename)) 
        return images;    
    
    def mean_image(self, images):
        #computes mean image of set IMAGES
        tot_image = np.zeros(images[0].shape)
        for image in images:
            tot_image = tot_image + image
        m_image = tot_image/len(images)
        #scale to 255
        m_image = np.uint8(np.around((m_image*255/m_image.max())))
        
        return m_image;

    def resize_images(self, images):
        new_images = []
        for image in images:
            new_images.append(self.resize(image))
        return new_images;
    
    def pad(self, image):
        pad_size = abs(max(image.shape)-min(image.shape))
        pad_image = np.pad(image, ((pad_size//2,pad_size//2),(0,0)), 'constant', constant_values=(0, 0))
        return pad_image;
    
    def pad_images(self, images):
        padded_images = []
        for image in images:
            padded_images.append(self.pad(image))
        return padded_images;
    
    def resize(self, image, size=64):
        #calculate the resize factor 
        rs_factor = size/max(image.shape)
        new_size  = tuple([int(round(num*rs_factor)) for num in image.shape])
        #use Image.resize function to convert 
        im = Image.fromarray(image)
        im = im.resize(new_size, Image.ANTIALIAS)
        #convert back to np array
        new_image = np.array(im)
        return new_image;
 
    def sub_mean_image(self, images, m_image):
        new_images = []
        for image in images:
            new_images.append(image - m_image)
        
        return new_images;

    def image2vector(self, image):
        v_image = np.reshape(image, tuple([1,np.prod(image.shape)]))
        return v_image;

    def vector2image(self, v_image, shape):
        image = np.reshape(v_image, shape)
        #scale to 255 uint
        #image = np.uint8(image*255/image.max())
        return image;
    
    def vectorize_images(self, images):
        v_images = []
        for image in images:
            v_images.append(self.image2vector(image))
        return v_images;
    
    def cat_vectors(self, images_v):
        image_samples = len(images_v)
        image_len     = images_v[0].shape[1]
        images_a = np.empty((image_samples, image_len))
        for k,image_v in enumerate(images_v):
            images_a[k,:] = image_v
        
        return images_a.astype('float')
    
    def normalize(self, image):
        n_image = (255*(image - np.max(image))/-np.ptp(image)).astype(int)
        
        return n_image;


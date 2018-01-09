# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import eigenfaces.face as face
import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')

fc = face.face_classifier()

fc.set_learning_set_a('images/learning/faces/')
fc.set_learning_set_b('images/learning/smiling_faces/')

images_s = fc.get_learning_set_a()

mean_s = fc.get_mean_a()

plt.figure()
plt.imshow(mean_s, cmap='gray')
#%%
eigen_faces_s = fc.get_eigenfaces_a()
eigen_faces_ns = fc.get_eigenfaces_b()

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
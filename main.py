# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import eigenfaces.face as face
import logging
import logging.config

logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger(__name__)
#logger_face = logging.getLogger('face')

logger.debug('Create classifier')
fc = face.face_classifier()

fc.set_learning_set_a('images/learning/faces/')
fc.set_learning_set_b('images/learning/smiling_faces/')

images_s = fc.get_learning_set_a()
logger.debug('get mean image')
mean_s = fc.get_mean_a()

plt.figure()
plt.imshow(mean_s, cmap='gray')
plt.title("Average Face")
#%%
eigen_faces_ns = fc.get_eigenfaces_a()
eigen_faces_s = fc.get_eigenfaces_b()

fig_s = plt.figure()

for k in range(16):
    plt.subplot(4,4,k+1)
    plt.axis('off')
    plt.imshow(eigen_faces_s[k], cmap='gray')
plt.suptitle("Smiling Eigenfaces")

fig_ns = plt.figure()
    
for k in range(16):
    plt.subplot(4,4,k+1)
    plt.axis('off')
    plt.imshow(eigen_faces_ns[k], cmap='gray')
plt.suptitle("Non-Smiling Eigenfaces")
# %% initilisation
import tensorflow   as tf
import niftynet     as nf

from niftynet               import layer
from niftynet.application   import classification_application

import matplotlib.pyplot    as plt
import numpy                as np
import random               as rd

print(tf.__version__)
print(nf.__version__)

# %% data

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(len(y_train))

# %% first image

img = rd.randint(0,len(y_train))
plt.figure()
plt.imshow(x_train[img])
plt.colorbar()
plt.grid(False)
plt.show()

# %% using niftynet

ClassAppli = nf.application.classification_application.ClassificationApplication

datas = ClassAppli
print(datas)

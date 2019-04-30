import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.path.join('~/travaux/traineeship/loss_function')

from cross_entropy import cross

learning_rate = tf.placeholder(tf.float64, shape=[])
adam = tf.train.AdamOptimizer(learning_rate)
pred = np.array([[0.2, 0.3, 1.4], [0.2, 0.4, 1.6], [1.2, 0.5, 0.6]])
labe = np.array([0, 1, 1])
weight = None

sess = tf.Session()
data = cross(pred, labe)
with sess.as_default():
    grad = adam.compute_gradients(data)
    print(grad)

# Affichage
# plt.plot(learning_rate, prompt, lw=1)
# plt.show()
# print('fin')

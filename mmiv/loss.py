import tensorflow as tf
from tensorflow.train import AdamOptimizer as Adam
from niftynet.layer.loss_segmentation import cross_entropy as cross
import numpy as np
import matplotlib.pyplot as plt

lr = 1e-3
arret = 2

prediction = tf.placeholder(tf.float64, shape=[2*10**6])
labels = np.linspace(0, 150, 2*10**6)%3
weight = np.logspace(0, 150, 2*10**6)

adam = Adam(learning_rate=lr)
loss = cross(prediction, labels, weight)
sess = tf.Session()
print(loss.eval(session=sess))

lr = 1e-6
while arret:
	adam = Adam(learning_rate=lr)
	print(adam.compute_gradients(loss, var_list=[prediction]))
	arret = False

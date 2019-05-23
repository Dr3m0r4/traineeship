import os
import tensorflow as tf
import numpy as np

import nibabel as nib

path = os.path.join("/home/julien/documents/results")
jeu = eval(input("enter the folder number : "))

labels = os.path.join(path, "input")
inference = os.path.join(path, "output_{}".format(jeu))

info = np.array(os.listdir(inference))
labo = np.array(os.listdir(labels))

n = len(info)
mara = np.array([])
for name in info:
	if name in labo:
		labf = os.path.join(labels, name)
		inff = os.path.join(inference, name)

		labd = nib.load(labf).get_data()
		infd = nib.load(inff).get_data()

		ind = np.where(labd==1)
		mara = np.append(mara, np.sum(infd[ind]==1)*2.0/(np.sum(infd==1)+np.sum(labd==1)))
mara = np.array(mara)

print("the dice result is :", np.sum(mara)/n)

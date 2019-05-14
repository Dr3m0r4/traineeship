import os
import tensorflow as tf
import numpy as np

import nibabel as nib

path = os.path.join("labels")
liste = os.listdir(path)
cpt = 0

for name in liste:
	cpt+=1
	file = os.path.join(path, name)
	file_bin = os.path.join('labels_bin', name)
	img = nib.load(file)
	data = img.get_data().copy()
	ind = np.where(data > 0)
	data_bin = np.zeros(data.shape)
	data_bin[ind]=1
	print(name," | iter : ", cpt)
	img_bin = nib.Nifti1Image(data_bin, img.affine)
	img_bin.header.set_data_dtype(np.float64)
	img_bin.header.set_xyzt_units('mm')
	nib.save(img_bin,file_bin)

# %% initilisation

import sys
sys.path.append('/home/dr3m/travaux/traineeship/niftynet_tutorial/demos')

import  os
import  nibabel             as nib
import  pandas              as pd
import  numpy               as np
import  matplotlib.pyplot   as plt
from mpl_toolkits.mplot3d   import Axes3D
from    skimage.io          import imread
import  re
import  seaborn             as sns
from unet_demo_utils        import *

from niftynet.io.image_reader                                   import ImageReader
from niftynet.contrib.sampler_pairwise.sampler_pairwise_uniform import PairwiseUniformSampler
from niftynet.layer.pad                                         import PadLayer
from niftynet.layer.rand_elastic_deform                         import RandomElasticDeformationLayer
from niftynet.layer.mean_variance_normalisation                 import MeanVarNormalisationLayer
from niftynet.layer.rand_flip                                   import RandomFlipLayer

import tensorflow as tf

print(tf._version_)

# %% definition function
def plot_slides(images, figsize=(10,5)):
    f, axes = plt.subplots(2,3, figsize=figsize)
    for i, slice_id in enumerate(images):
        axes[i][0].imshow(images[slice_id]['img'], cmap='gray')
        axes[i][0].set_title('Image %s' % slice_id)
        axes[i][1].imshow(images[slice_id]['seg'], cmap='gray')
        axes[i][1].set_title('Segmentation %s' % slice_id)
        axes[i][2].imshow(images[slice_id]['weight'], cmap='jet', vmin=0, vmax=10)
        axes[i][2].set_title('Weight Map %s' % slice_id)

        for ax in axes[i]:
            ax.set_axis_off()
    f.tight_layout()


def grab_demo_images(image_dir, slice_ids, image_prefix_dict):
    images = {slice_id: {
            key: imread(os.path.join(image_dir, image_prefix_dict[key] + '%s.tif' % slice_id))
            for key in image_prefix_dict}
        for slice_id in slice_ids}
    return images

def create_image_reader(num_controlpoints, std_deformation_sigma):
    # creating an image reader.
    data_param = \
        {'cell': {'path_to_search': '~/travaux/traineeship/niftynet_tutorial/demos/data/PhC-C2DH-U373/niftynet_data', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'img_',
                'loader': 'skimage'},
         'label': {'path_to_search': '~/travaux/traineeship/niftynet_tutorial/demos/data/PhC-C2DH-U373/niftynet_data', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'bin_seg_',
                'loader': 'skimage',
                'interp_order' : 0}
        }
    reader = ImageReader().initialise(data_param)

    reader.add_preprocessing_layers(MeanVarNormalisationLayer(image_name = 'cell'))

    reader.add_preprocessing_layers(PadLayer(
                     image_name=['cell', 'label'],
                     border=(92,92,0),
                     mode='symmetric'))

    reader.add_preprocessing_layers(RandomElasticDeformationLayer(
                     num_controlpoints=num_controlpoints,
                     std_deformation_sigma=std_deformation_sigma,
                     proportion_to_augment=1,
                     spatial_rank=2))

#     reader.add_preprocessing_layers(RandomFlipLayer(
#                  flip_axes=(0,1)))

    return reader

# %%

U373_dir = "niftynet_tutorial/demos/data/PhC-C2DH-U373/niftynet_data"
U373_imgs = grab_demo_images(U373_dir, ['049_01', '049_02'], {'img': 'img_', 'seg': 'bin_seg_', 'weight': 'weight_'})

plot_slides(U373_imgs, figsize=(9,5))

# %%

HeLa_dir = "niftynet_tutorial/demos/data/DIC-C2DH-HeLa/niftynet_data/"
HeLa_images = grab_demo_images(HeLa_dir, ['067_01', '038_02'], {'img': 'img_', 'seg': 'bin_seg_', 'weight':'weight_'})

plot_slides(HeLa_images, figsize=(9, 7))

# %%



f, axes = plt.subplots(5,4,figsize=(15,15))
f.suptitle('The same input image, deformed under varying $\sigma$')

for i, axe in enumerate(axes):
    std_sigma = 25 * i
    reader = create_image_reader(6, std_sigma)
    for ax in axe:
        _, image_data, _ = reader(1)
        ax.imshow(image_data['cell'].squeeze(), cmap='gray')
        ax.imshow(image_data['label'].squeeze(), cmap='jet', alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Deformation Sigma = %i' % std_sigma)

# %%

data_dir = "/home/dr3m/travaux/traineeship/niftynet_tutorial/demos/data/PhC-C2DH-U373/niftynet_data/"

u373_ground_truths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith('bin_seg')]
est_dirs = {x: "../../models/U373_" + str(x) + "/output/" for x in range(8)}
u373_ids = ('001_01', '059_02')

df_u373 = get_and_plot_results(u373_ground_truths, est_dirs, u373_ids)

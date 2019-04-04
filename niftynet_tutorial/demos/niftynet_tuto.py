# %% initilisation
import  os
import  nibabel             as nib
import  pandas              as pd
import  numpy               as np
import  matplotlib.pyplot   as plt
from mpl_toolkits.mplot3d   import Axes3D
from    skimage.io          import imread
import  re
import  seaborn             as sns

from niftynet.io.image_reader                                   import ImageReader
from niftynet.contrib.sampler_pairwise.sampler_pairwise_uniform import PairwiseUniformSampler
from niftynet.layer.pad                                         import PadLayer
from niftynet.layer.rand_elastic_deform                         import RandomElasticDeformationLayer
from niftynet.layer.mean_variance_normalisation                 import MeanVarNormalisationLayer
from niftynet.layer.rand_flip                                   import RandomFlipLayer

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
        {'cell': {'path_to_search': '~/niftynet/data/dense_vnet_abdominal_ct', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'CT',
                'loader': 'skimage'},
         'label': {'path_to_search': '~/niftynet/data/dense_vnet_abdominal_ct', # PhC-C2DH-U373, DIC-C2DH-HeLa
                'filename_contains': 'Label',
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

f, axes = plt.subplots(5,4,figsize=(15,15))
f.suptitle('The same input image, deformed under varying $\sigma$')

for i, axe in enumerate(axes):
    std_sigma = 25 * i
    reader = create_image_reader(6, std_sigma)
    for ax in axe:
        _, image_data, _  = reader(1)

        ax.imshow(image_data['cell'].squeeze()[:,:,0], cmap='gray')
        ax.imshow(image_data['label'].squeeze()[:,:,0], cmap='jet', alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Deformation Sigma = %i' % std_sigma)
plt.show()

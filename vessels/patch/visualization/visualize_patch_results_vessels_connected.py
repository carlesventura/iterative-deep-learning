__author__ = 'carlesv'
import matplotlib.pyplot as plt
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np


dist_th = 4
num_images = 20
num_patches = 50
start_img = 1
patch_size = 64

#peak_th = 100

gt_base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test'
results_base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results'


idx_patch = 2


#for idx in range(1,num_images+1):
for idx in range(start_img,start_img+num_images):

    print(idx)

    config_type = '_connected'
    retina_img = Image.open(gt_base_dir + config_type + '/img_%02d_patch_%02d_img.png' %(idx, idx_patch))
    plt.imshow(retina_img)
    pred = np.load(results_base_dir + config_type + '/epoch_1800/img_%02d_patch_%02d.npy' %(idx, idx_patch))

    mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
    threshold = median + (10.0 * std)
    sources = find_peaks(pred, threshold, box_size=3)
    positions = (sources['x_peak'], sources['y_peak'])

    pos_x_vector = []
    pos_y_vector = []
    for ii in range(0,len(sources['peak_value'])):
        if sources['peak_value'][ii] > 20:
            pos_x_vector.append(sources['x_peak'][ii])
            pos_y_vector.append(sources['y_peak'][ii])

    plt.scatter(pos_x_vector, pos_y_vector, marker='+', color='blue', s=100, linewidths=10)
    plt.axis('off')
    plt.show()
    #axes[2,config_id].plot(sources['x_peak'], sources['y_peak'], ls='none', color='red',marker='+', ms=10, lw=1.5)










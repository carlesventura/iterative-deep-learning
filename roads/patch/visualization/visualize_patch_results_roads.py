__author__ = 'carlesv'
import matplotlib.pyplot as plt
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np

gt_base_dir = './results_dir/val_gt/'
results_base_dir = './results_dir/val_results/'

num_images = 14
num_patches = 50
start_img = 1
patch_size = 64

for idx_patch in range(1,num_patches+1):
    for idx in range(start_img,start_img+num_images):

        road_img = Image.open(gt_base_dir + 'img_%02d_patch_%02d_img.png' %(idx, idx_patch))
        plt.imshow(road_img)
        pred = np.load(results_base_dir + 'img_%02d_patch_%02d.npy' %(idx, idx_patch))

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

        plt.scatter(pos_x_vector, pos_y_vector, marker='+', color='red', s=100, linewidth=3)
        plt.axis('off')
        plt.show()


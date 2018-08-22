__author__ = 'carlesv'

import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


img_idx = 1
#ii=101
ii = 1
previous_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_iter_%05d.png' % (img_idx,ii))
#previous_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_with_novelty_iter_%05d.png' % (img_idx,ii))
#previous_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_only_novelty_iter_%05d.png' % (img_idx,ii))
previous_mask = np.array(previous_mask)

root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))

while ii < 20000:

    ii = ii + 100
    print(ii)
    plt.imshow(img)
    new_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_iter_%05d.png' % (img_idx,ii))
    #new_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_with_novelty_iter_%05d.png' % (img_idx,ii))
    #new_mask = Image.open('/scratch_net/boxy/carlesv/results/DRIVE/tmp_results/pred_graph_%02d_mask_same_mask_graph_th_30_only_novelty_iter_%05d.png' % (img_idx,ii))
    new_mask = np.array(new_mask)

    indxs_previous = np.argwhere(previous_mask==255)
    plt.scatter(indxs_previous[:,1],indxs_previous[:,0],color='green',marker='+')

    same_mask =  new_mask==previous_mask
    indxs_diff = np.argwhere(same_mask==0)
    plt.scatter(indxs_diff[:,1],indxs_diff[:,0],color='red',marker='+')
    plt.show()

    previous_mask = new_mask

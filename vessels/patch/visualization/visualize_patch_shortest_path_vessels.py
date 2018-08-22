__author__ = 'carlesv'
from PIL import Image
import torch
from torch.autograd import Variable
import Nets as nt
import os
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np
import vessels.patch.bifurcations_toolbox as tb
from astropy.table import Table
import matplotlib.pyplot as plt
import vessels.iterative.shortest_path as sp
import networkx as nx



def get_most_confident_outputs(img_id, patch_center_row, patch_center_col, confident_th, gpu_id, connected_same_vessel):

    patch_size = 64
    center = (patch_center_col, patch_center_row)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    confident_connections = {}
    confident_connections['x_peak'] = []
    confident_connections['y_peak'] = []
    confident_connections['peak_value'] = []

    root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_id))
    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    if x_tmp > 0 and y_tmp > 0 and x_tmp+patch_size < w and y_tmp+patch_size < h:

        img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

        img_crop = img_crop.transpose((2, 0, 1))
        img_crop = torch.from_numpy(img_crop)
        img_crop = img_crop.unsqueeze(0)

        inputs = img_crop / 255 - 0.5

        # Forward pass of the mini-batch
        inputs = Variable(inputs)

        #gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
        #gpu_id = -1
        if gpu_id >= 0:
            #torch.cuda.set_device(device=gpu_id)
            inputs = inputs.cuda()

        p = {}
        p['useRandom'] = 1  # Shuffle Images
        p['useAug'] = 0  # Use Random rotations in [-30, 30] and scaling in [.75, 1.25]
        p['inputRes'] = (64, 64)  # Input Resolution
        p['outputRes'] = (64, 64)  # Output Resolution (same as input)
        p['g_size'] = 64  # Higher means narrower Gaussian
        p['trainBatch'] = 1  # Number of Images in each mini-batch
        p['numHG'] = 2  # Number of Stacked Hourglasses
        p['Block'] = 'ConvBlock'  # Select: 'ConvBlock', 'BasicBlock', 'BottleNeck'
        p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images
        model_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/'
        if connected_same_vessel:
            modelName = tb.construct_name(p, "HourGlass-connected-same-vessel")
        else:
            modelName = tb.construct_name(p, "HourGlass-connected")
        #modelName = tb.construct_name(p, "HourGlass-connected-same-vessel-wo-bifurcations")
        numHGScales = 4  # How many times to downsample inside each HourGlass
        net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
        epoch = 1800
        net.load_state_dict(torch.load(os.path.join(model_dir, os.path.join(model_dir, modelName+'_epoch-'+str(epoch)+'.pth')),
                                   map_location=lambda storage, loc: storage))

        if gpu_id >= 0:
            net = net.cuda()

        output = net.forward(inputs)
        pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))


        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)

        indxs = np.argsort(sources['peak_value'])
        for ii in range(0,len(indxs)):
            idx = indxs[len(indxs)-1-ii]
            if sources['peak_value'][idx] > confident_th:
                confident_connections['x_peak'].append(sources['x_peak'][idx])
                confident_connections['y_peak'].append(sources['y_peak'][idx])
                confident_connections['peak_value'].append(sources['peak_value'][idx])
            else:
                break

        confident_connections = Table([confident_connections['x_peak'], confident_connections['y_peak'], confident_connections['peak_value']], names=('x_peak', 'y_peak', 'peak_value'))

    return confident_connections




img_idx = 1

root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))

plt.imshow(img)
plt.show()

img_array = np.array(img, dtype=np.uint8)
h, w = img_array.shape[:2]

start_row = 470
start_col = 396



patch_size = 64
x_tmp = int(start_col-patch_size/2)
y_tmp = int(start_row-patch_size/2)
img_crop = img_array[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

center = (start_col, start_row)

confident_connections = get_most_confident_outputs(img_idx, start_row, start_col, 25, -1, False)
G = sp.generate_graph_center(img_idx,center)

plt.imshow(img_crop)

target_idx = 32*64 + 32
for ii in range(0,len(confident_connections['y_peak'])):
    source_idx = (confident_connections['y_peak'][ii])*64 + confident_connections['x_peak'][ii]
    length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
    pos_y_vector = []
    pos_x_vector = []
    for jj in range(0,len(path)):
        row_idx = path[jj] / 64
        col_idx = path[jj] % 64
        #global_x = col_idx+start_col-32
        #global_y = row_idx+start_row-32
        pos_y_vector.append(row_idx)
        pos_x_vector.append(col_idx)
    plt.scatter(pos_x_vector, pos_y_vector, marker='+', color='green', s=100, linewidth=3)
plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
plt.show()




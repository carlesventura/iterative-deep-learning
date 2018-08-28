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
from scipy import ndimage


def get_most_confident_outputs(img_id, patch_center_row, patch_center_col, confident_th, gpu_id, connected_same_vessel):

    patch_size = 64
    center = (patch_center_col, patch_center_row)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    confident_connections = {}
    confident_connections['x_peak'] = []
    confident_connections['y_peak'] = []
    confident_connections['peak_value'] = []

    root_dir = './gt_dbs/DRIVE/'
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

        if gpu_id >= 0:
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
        model_dir = './results_dir_vessels/'
        if connected_same_vessel:
            modelName = tb.construct_name(p, "HourGlass-connected-same-vessel")
        else:
            modelName = tb.construct_name(p, "HourGlass-connected")
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






start_row_all = np.array([312, 312, 330, 270, 330, 336, 349, 216, 348, 351, 325, 322, 342, 359, 318, 337, 348, 373, 349, 367])
start_col_all = np.array([91, 442, 99, 297, 100, 476, 442, 451, 109, 462, 99, 98, 489, 495, 242, 461, 449, 446, 491, 458])


connected_same_vessel = False
confident_th = 30

#gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
gpu_id = -1
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)


for img_idx in range(1,21):

    start_row = start_row_all[img_idx-1]
    start_col = start_col_all[img_idx-1]

    root_dir = './gt_dbs/DRIVE/'
    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))

    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    center_points = []
    center = (start_col, start_row)
    center_points.append(center)

    colors = ['red', 'blue', 'cyan', 'green', 'purple']

    mask = np.zeros((h,w))
    mask_outputs = np.zeros((h,w))
    mask_iter = np.zeros((h,w))

    pending_connections_x = []
    pending_connections_y = []
    parent_connections_x = []
    parent_connections_y = []

    confidence_pending_connections = []
    pending_connections_x.append(center[0])
    pending_connections_y.append(center[1])
    confidence_pending_connections.append(255)
    mask_graph = np.zeros((h,w))

    connections_to_be_extended = []
    connections_to_be_extended.append(True)

    parent_connections_x.append(center[0])
    parent_connections_y.append(center[1])

    offset = 0
    offset_mask = 0
    mask[center[1]-offset_mask:center[1]+offset_mask+1,center[0]-offset_mask:center[0]+offset_mask+1] = 1
    mask_outputs[center[1]-offset_mask:center[1]+offset_mask+1,center[0]-offset_mask:center[0]+offset_mask+1] = 1

    ii = 0
    while len(pending_connections_x) > 0:
        max_idx = np.argmax(confidence_pending_connections)
        next_element_x = pending_connections_x[max_idx]
        next_element_y = pending_connections_y[max_idx]

        if ii > 0:

            previous_center_x = parent_connections_x[max_idx]
            previous_center_y = parent_connections_y[max_idx]

            tmp_center = (previous_center_x,previous_center_y)
            G = sp.generate_graph_center(img_idx,tmp_center)
            target_idx = 32*64 + 32
            source_idx = (next_element_y-previous_center_y+32)*64 + next_element_x-previous_center_x+32
            length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path)):
                row_idx = path[jj] / 64
                col_idx = path[jj] % 64
                global_x = col_idx+previous_center_x-32
                global_y = row_idx+previous_center_y-32
                if mask_graph[global_y,global_x] == 0:
                    pos_y_vector.append(global_y)
                    pos_x_vector.append(global_x)
                else:
                    break


            if len(pos_y_vector) > 0:

                for kk in range(0,len(pos_y_vector)):
                    mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                    mask[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                    mask_iter[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = ii


            #Do the same but from target (center) to source (connected output)
            target_idx = source_idx
            source_idx = 32*64 + 32
            length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path)):
                row_idx = path[jj] / 64
                col_idx = path[jj] % 64
                global_x = col_idx+previous_center_x-32
                global_y = row_idx+previous_center_y-32
                if mask_graph[global_y,global_x] == 0:
                    pos_y_vector.append(global_y)
                    pos_x_vector.append(global_x)
                else:
                    break

            if len(pos_y_vector) > 0:

                for kk in range(0,len(pos_y_vector)):
                    mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                    mask[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                    mask_iter[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = ii


        plt.imshow(img)
        indxs = np.argwhere(mask_graph==1)
        plt.scatter(indxs[:,1],indxs[:,0],color='green',marker='+')
        plt.show()



        confidence_pending_connections = np.delete(confidence_pending_connections,max_idx)
        pending_connections_x = np.delete(pending_connections_x,max_idx)
        pending_connections_y = np.delete(pending_connections_y,max_idx)

        parent_connections_x = np.delete(parent_connections_x,max_idx)
        parent_connections_y = np.delete(parent_connections_y,max_idx)

        to_be_extended = connections_to_be_extended[max_idx]
        connections_to_be_extended = np.delete(connections_to_be_extended,max_idx)

        if to_be_extended:

            confident_connections = get_most_confident_outputs(img_idx, next_element_y, next_element_x, confident_th, gpu_id, connected_same_vessel)

            for kk in range(0,len(confident_connections['peak_value'])):
                tmp_x = confident_connections['x_peak'][kk]+next_element_x-32
                tmp_y = confident_connections['y_peak'][kk]+next_element_y-32

                if mask[tmp_y,tmp_x] == 0:

                    pending_connections_x = np.append(pending_connections_x,tmp_x)
                    pending_connections_y = np.append(pending_connections_y,tmp_y)
                    confidence_pending_connections = np.append(confidence_pending_connections,confident_connections['peak_value'][kk])

                    parent_connections_x = np.append(parent_connections_x,next_element_x)
                    parent_connections_y = np.append(parent_connections_y,next_element_y)

                    min_y = np.max([0, tmp_y-offset_mask])
                    min_x = np.max([0, tmp_x-offset_mask])
                    max_y = np.min([h-1, tmp_y+offset_mask+1])
                    max_x = np.min([w-1, tmp_x+offset_mask+1])
                    mask[min_y:max_y,min_x:max_x] = 1
                    mask_outputs[min_y:max_y,min_x:max_x] = 1

                    connections_to_be_extended = np.append(connections_to_be_extended,True)

                else:

                    pending_connections_x = np.append(pending_connections_x,tmp_x)
                    pending_connections_y = np.append(pending_connections_y,tmp_y)
                    confidence_pending_connections = np.append(confidence_pending_connections,confident_connections['peak_value'][kk])

                    parent_connections_x = np.append(parent_connections_x,next_element_x)
                    parent_connections_y = np.append(parent_connections_y,next_element_y)

                    min_y = np.max([0, tmp_y-offset_mask])
                    min_x = np.max([0, tmp_x-offset_mask])
                    max_y = np.min([h-1, tmp_y+offset_mask+1])
                    max_x = np.min([w-1, tmp_x+offset_mask+1])
                    mask_outputs[min_y:max_y,min_x:max_x] = 1

                    connections_to_be_extended = np.append(connections_to_be_extended,False)


        ii += 1

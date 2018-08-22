__author__ = 'carlesv'
import numpy as np
import scipy.io as sio
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import vessels.iterative.shortest_path as sp
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage
from astropy.table import Table
import os
import torch
from torch.autograd import Variable
import Nets as nt
import vessels.patch.bifurcations_toolbox as tb

idx = 1
idx_patch = 50
num_images = 20

DRIU_baseline = False
connected = True
from_same_vessel = False
bifurcations_allowed = True

num_outputs = 6


def valid_sources(sources):
    samples = np.zeros((len(sources),2),int)
    for ii in range(0,len(sources)):
        samples[ii,0] = sources['x_peak'][ii]
        samples[ii,1] = sources['y_peak'][ii]

    Z = linkage(samples, 'single', 'cityblock')
    max_d = 1
    clusters = fcluster(Z, max_d, criterion='distance')
    clusters_visited = []
    clustered_sources = {}
    clustered_sources['x_peak'] = []
    clustered_sources['y_peak'] = []
    clustered_sources['peak_value'] = []

    for ii in range(0, len(clusters)):
        cluster_idx = clusters[ii]
        if cluster_idx not in clusters_visited:
            clusters_visited.append(cluster_idx)
            sample_idxs = np.argwhere(clusters == cluster_idx)
            mean_x = np.mean(sources['x_peak'][sample_idxs])
            mean_y = np.mean(sources['y_peak'][sample_idxs])
            mean_peak = np.mean(sources['peak_value'][sample_idxs])
            clustered_sources['x_peak'].append(mean_x)
            clustered_sources['y_peak'].append(mean_y)
            clustered_sources['peak_value'].append(mean_peak)

    clustered_sources = Table([clustered_sources['x_peak'], clustered_sources['y_peak'], clustered_sources['peak_value']], names=('x_peak', 'y_peak', 'peak_value'))

    return clustered_sources

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_most_confident_outputs(img_id, patch_center_row, patch_center_col, confident_th):

    patch_size = 64
    center = (patch_center_col, patch_center_row)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_id))
    img = np.array(img, dtype=np.float32)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    img_crop = img_crop.transpose((2, 0, 1))
    img_crop = torch.from_numpy(img_crop)
    img_crop = img_crop.unsqueeze(0)

    inputs = img_crop / 255 - 0.5

    # Forward pass of the mini-batch
    inputs = Variable(inputs)

    gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
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
    model_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/'
    modelName = tb.construct_name(p, "HourGlass-connected")
    numHGScales = 4  # How many times to downsample inside each HourGlass
    net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
    epoch = 1800
    net.load_state_dict(torch.load(os.path.join(model_dir, os.path.join(model_dir, modelName+'_epoch-'+str(epoch)+'.pth')),
                               map_location=lambda storage, loc: storage))

    output = net.forward(inputs)
    pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))


    mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
    threshold = median + (10.0 * std)
    sources = find_peaks(pred, threshold, box_size=3)

    confident_connections = {}
    confident_connections['x_peak'] = []
    confident_connections['y_peak'] = []
    confident_connections['peak_value'] = []

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

def get_graph_from_patch(img_id, patch_center_row, patch_center_col):

    patch_size = 64
    center = (patch_center_col, patch_center_row)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_id))
    img = np.array(img, dtype=np.float32)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    img_crop = img_crop.transpose((2, 0, 1))
    img_crop = torch.from_numpy(img_crop)
    img_crop = img_crop.unsqueeze(0)

    inputs = img_crop / 255 - 0.5

    # Forward pass of the mini-batch
    inputs = Variable(inputs)

    gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
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
    model_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/'
    modelName = tb.construct_name(p, "HourGlass-connected")
    numHGScales = 4  # How many times to downsample inside each HourGlass
    net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
    epoch = 1800
    net.load_state_dict(torch.load(os.path.join(model_dir, os.path.join(model_dir, modelName+'_epoch-'+str(epoch)+'.pth')),
                               map_location=lambda storage, loc: storage))

    output = net.forward(inputs)
    pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))


    mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
    threshold = median + (10.0 * std)
    sources = find_peaks(pred, threshold, box_size=3)

    target_row = 31
    target_col = 31
    target_idx = target_row*pred.shape[1] + target_col

    G = sp.generate_graph_center(img_id, center)
    visited_nodes = []
    visited_nodes.append(target_idx)

    indxs = np.argsort(sources['peak_value'])
    count_confident_connections = 0
    confident_th = 15
    for ii in range(0,len(indxs)):
        idx = indxs[len(indxs)-1-ii]
        if sources['peak_value'][idx] > confident_th:
            count_confident_connections += 1
        else:
            break

    target_reached = []
    points_from_lines = []
    parent_line = []
    parent_point = []

    for ii in range(0,count_confident_connections):
        points_from_current_line = []
        idx = indxs[len(indxs)-1-ii]
        source_col_idx = sources['x_peak'][idx]
        source_row_idx = sources['y_peak'][idx]
        source_idx = source_row_idx*pred.shape[1] + source_col_idx

        length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
        color = np.random.rand(3,)
        for jj in range(0,len(path)):
            if path[jj] not in visited_nodes:
                #Add node to current line
                points_from_current_line.append(path[jj])
                visited_nodes.append(path[jj])

            else:
                #Junction found
                points_from_lines.append(points_from_current_line)
                if path[jj] == target_idx:
                    target_reached.append(ii)
                    parent_line.append(-1)
                    parent_point.append(-1)
                else:
                    parent_point.append(path[jj])
                    for kk in range(0,ii):
                        if path[jj] in points_from_lines[kk]:
                            parent_line.append(kk)
                            break
                break

    return points_from_lines, parent_line, parent_point



if DRIU_baseline:
    results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/'
else:
    if not connected:
        results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_not_connected/'
    else:
        if from_same_vessel:
            if bifurcations_allowed:
                results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected_same_vessel/'
            else:
                results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected_same_vessel_wo_bifurcations/'
        else:
            results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected/'

pred_vessels = Image.open('/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/%02d_test.png' %(idx))
pred_vessels = np.array(pred_vessels)

f = open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/vertices_selected.txt','r')
count = 0

while count != (idx_patch-1)*num_images + idx-1:
    line = f.readline()
    count += 1

line = f.readline()
f.close()

selected_vertex = int(line.split()[1])

mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

patch_size = 64
x_tmp = int(center[0]-patch_size/2)
y_tmp = int(center[1]-patch_size/2)

pred_vessels = pred_vessels[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

if DRIU_baseline:
    # pred = Image.open(results_dir + '%02d_test.png' %(idx))
    # pred = np.array(pred)
    #
    # f = open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/vertices_selected.txt','r')
    # count = 0
    #
    # while count != (idx_patch-1)*num_images + idx-1:
    #     line = f.readline()
    #     count += 1
    #
    # line = f.readline()
    # f.close()
    #
    # selected_vertex = int(line.split()[1])
    #
    # mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    # vertices = np.squeeze(mat_contents['G']['V'][0,0])
    # center = (vertices[selected_vertex,0], vertices[selected_vertex,1])
    #
    # patch_size = 64
    # x_tmp = int(center[0]-patch_size/2)
    # y_tmp = int(center[1]-patch_size/2)
    #
    # pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]
    pred = pred_vessels
    margin = int(np.round(patch_size/10.0))
    pred[0:margin,:] = 0
    pred[margin:patch_size-margin,0:margin] = 0
    pred[margin:patch_size-margin,patch_size-margin:patch_size] = 0
    pred[patch_size-margin:patch_size,:] = 0
    pred[margin+1:patch_size-margin-1,margin+1:patch_size-margin-1] = 0


else:
    pred = np.load(results_dir + 'epoch_1800/img_%02d_patch_%02d.npy' %(idx, idx_patch))

mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
threshold = median + (10.0 * std)
sources = find_peaks(pred, threshold, box_size=3)

if DRIU_baseline:
    sources = valid_sources(sources)



target_row = 31
target_col = 31
target_idx = target_row*pred.shape[1] + target_col

G = sp.generate_graph(idx, idx_patch)
visited_nodes = []
visited_nodes.append(target_idx)

#fig, axes = plt.subplots(2, 3)
#axes[0,0].imshow(pred_vessels)
plt.imshow(pred_vessels)

indxs = np.argsort(sources['peak_value'])

target_reached = []
points_from_lines = []
parent_line = []
parent_point = []

cmap = get_cmap(num_outputs)

#for ii in range(0,len(indxs)):
for ii in range(0,num_outputs):
    points_from_current_line = []
    idx = indxs[len(indxs)-1-ii]
    source_col_idx = sources['x_peak'][idx]
    source_row_idx = sources['y_peak'][idx]
    source_idx = source_row_idx*pred.shape[1] + source_col_idx

    length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
    color = np.random.rand(3,)
    for jj in range(0,len(path)):
        if path[jj] not in visited_nodes:
            #Add node to current line
            points_from_current_line.append(path[jj])
            visited_nodes.append(path[jj])
            row_idx = path[jj] / pred.shape[1]
            col_idx = path[jj] % pred.shape[1]
            #axes[0,0].plot(col_idx,row_idx,color='red',marker='+')
            if jj == 0:
                #axes[(ii+1)/3,(ii+1)%3].plot(col_idx,row_idx,color=color,marker='o')
                plt.plot(col_idx,row_idx,color=color,marker='o')
            else:
                #axes[(ii+1)/3,(ii+1)%3].plot(col_idx,row_idx,color=color,marker='+')
                plt.plot(col_idx,row_idx,color=color,marker='+')
            #axes[1].plot(col_idx,row_idx,color=cmap(ii),marker='+')
            #axes[(ii+1)/3,(ii+1)%3].set_aspect(1)
            #axes[(ii+1)/3,(ii+1)%3].set_xlim([0,63])
            #axes[(ii+1)/3,(ii+1)%3].set_ylim([63,0])
        else:
            #Junction found
            points_from_lines.append(points_from_current_line)
            print(points_from_current_line)
            if path[jj] == target_idx:
                target_reached.append(ii)
                parent_line.append(-1)
                parent_point.append(-1)
            else:
                parent_point.append(path[jj])
                for kk in range(0,ii):
                    if path[jj] in points_from_lines[kk]:
                        parent_line.append(kk)
                        break
            break

print(target_reached)
print(parent_line)
print(parent_point)
plt.show()

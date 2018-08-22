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
import shortest_path as sp
import networkx as nx
from scipy import ndimage
import scipy.misc


def get_most_confident_outputs(img_id, patch_center_row, patch_center_col, confident_th, gpu_id):

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

def get_most_confident_outputs_vessel_width(img_id, patch_center_row, patch_center_col, confident_th, gpu_id):

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
        modelName = tb.construct_name(p, "HourGlass-connected")
        #modelName = tb.construct_name(p, "HourGlass-connected-same-vessel-wo-bifurcations")
        numHGScales = 4  # How many times to downsample inside each HourGlass
        net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
        epoch = 1800
        net.load_state_dict(torch.load(os.path.join(model_dir, os.path.join(model_dir, modelName+'_epoch-'+str(epoch)+'.pth')),
                                   map_location=lambda storage, loc: storage))

        if gpu_id >= 0:
            net.cuda()

        output = net.forward(inputs)
        pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))


        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)

        results_dir_vessels = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/'

        pred_vessels = Image.open(results_dir_vessels + '%02d_test.png' %(img_id))
        pred_vessels = np.array(pred_vessels)
        pred_vessels = pred_vessels[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]


        for ii in range(0,len(sources['peak_value'])):

            mask = np.zeros((patch_size,patch_size))
            mask[int(sources['y_peak'][ii]),int(sources['x_peak'][ii])] = 1
            mask = ndimage.grey_dilation(mask, size=(5,5))
            pred_vessels_masked = np.ma.masked_array(pred_vessels, mask=(mask == 0))
            confidence_width = pred_vessels_masked.sum()
            #sources['peak_value'][ii] = sources['peak_value'][ii] + confidence_width
            if sources['peak_value'][ii] > confident_th:
                sources['peak_value'][ii] = confidence_width


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

visualize_graph = False

gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
#gpu_id = -1
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)

#img_idx = 1

for img_idx in range(1,11):

    #start_row = 312
    #start_col = 91

    start_row = start_row_all[img_idx-1]
    start_col = start_col_all[img_idx-1]

    center_optical_disk_row = 260.8
    center_optical_disk_col = 86.6

    radius_optical_disk = 40

    root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))
    if visualize_graph:
        import matplotlib.pyplot as plt
        #plt.imshow(img)

    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    center_points = []
    center = (start_col, start_row)
    center_points.append(center)
    confident_th = 15

    #colors = ['red', 'blue', 'black', 'green', 'purple']
    colors = ['red', 'blue', 'cyan', 'green', 'purple']

    mask = np.zeros((h,w))

    breadth_first_search = False

    segments = []

    if breadth_first_search: #old implementation

        for ii in range(0,5):
            new_center_points = []
            for jj in range(0,len(center_points)):
                #if visualize_graph:
                    #plt.plot(center_points[jj][0],center_points[jj][1],marker='+',color=colors[ii])
                confident_connections = get_most_confident_outputs(img_idx, center_points[jj][1], center_points[jj][0], confident_th, gpu_id)
                for kk in range(0,len(confident_connections)):
                    tmp_point = (center_points[jj][0]-32+confident_connections['x_peak'][kk],center_points[jj][1]-32+confident_connections['y_peak'][kk])
                    dist = (tmp_point[0]-center_optical_disk_col)*(tmp_point[0]-center_optical_disk_col) + (tmp_point[1]-center_optical_disk_row)*(tmp_point[1]-center_optical_disk_row)
                    if dist > radius_optical_disk*radius_optical_disk and mask[tmp_point[1],tmp_point[0]] == 0:
                        #if visualize_graph:
                            #plt.plot(tmp_point[0],tmp_point[1],marker='o',color=colors[ii], ms=10, lw=1.5, mfc='none')
                        new_center_points.append(tmp_point)

                mask[center_points[jj][1]-32:center_points[jj][1]+32,center_points[jj][0]-32:center_points[jj][0]+32] = 1

            center_points = new_center_points

    else: #most confident outputs are first visited
        pending_connections_x = []
        pending_connections_y = []
        parent_connections_x = []
        parent_connections_y = []

        confidence_pending_connections = []
        pending_connections_x.append(center[0])
        pending_connections_y.append(center[1])
        confidence_pending_connections.append(255)
        mask_graph = np.zeros((h,w))
        segment_ids = np.zeros((h,w),int)

        parent_connections_x.append(center[0])
        parent_connections_y.append(center[1])

        offset = 0
        mask[center[1]-offset:center[1]+offset+1,center[0]-offset:center[0]+offset+1] = 1

        count = 0
        #for ii in range(0,10000):
        ii = 0
        while len(pending_connections_x) > 0:
            max_idx = np.argmax(confidence_pending_connections)
            next_element_x = pending_connections_x[max_idx]
            next_element_y = pending_connections_y[max_idx]

            if ii > 0:

                previous_center_x = parent_connections_x[max_idx]
                previous_center_y = parent_connections_y[max_idx]

                #print([next_element_x, next_element_y])
                #print([previous_center_x, previous_center_y])

                tmp_center = (previous_center_x,previous_center_y)
                G = sp.generate_graph_center_patch_size_min_confidence(img_idx,tmp_center,64,50)
                #G = sp.generate_graph_center_connectivity4(img_idx,tmp_center)
                target_idx = 32*64 + 32
                source_idx = (next_element_y-previous_center_y+32)*64 + next_element_x-previous_center_x+32
                length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
                pos_y_vector = []
                pos_x_vector = []
                count_mask_graph = 0
                for jj in range(0,len(path)):
                    row_idx = path[jj] / 64
                    col_idx = path[jj] % 64
                    global_x = col_idx+previous_center_x-32
                    global_y = row_idx+previous_center_y-32
                    if mask_graph[global_y,global_x] == 0 or count_mask_graph < 0:
                        pos_y_vector.append(global_y)
                        pos_x_vector.append(global_x)
                        if mask_graph[global_y,global_x] == 1:
                            count_mask_graph += 1
                        #mask_graph[global_y,global_x] = 1
                        #plt.plot(global_x,global_y,color=colors[ii%5],marker='+')
                    else:
                        break


                if len(pos_y_vector) > 0:
                    new_segment = []
                    #if visualize_graph:
                        #plt.plot(next_element_x,next_element_y,marker='o',color=colors[count%5])
                    for kk in range(0,len(pos_y_vector)):
                        #mask_graph[pos_y_vector[kk],pos_x_vector[kk]] = 1
                        mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                        new_segment.append(pos_y_vector[kk]*w+pos_x_vector[kk])
                        segment_ids[pos_y_vector[kk],pos_x_vector[kk]] = count + 1
                    #if visualize_graph:
                        #plt.scatter(pos_x_vector,pos_y_vector,color=colors[count%5],marker='+')
                    count += 1
                    segments.append(new_segment)

                #Do the same but from target (center) to source (connected output)
                target_idx = source_idx
                source_idx = 32*64 + 32
                length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
                pos_y_vector = []
                pos_x_vector = []
                count_mask_graph = 0
                for jj in range(0,len(path)):
                    row_idx = path[jj] / 64
                    col_idx = path[jj] % 64
                    global_x = col_idx+previous_center_x-32
                    global_y = row_idx+previous_center_y-32
                    if mask_graph[global_y,global_x] == 0 or count_mask_graph < 0:
                        pos_y_vector.append(global_y)
                        pos_x_vector.append(global_x)
                        if mask_graph[global_y,global_x] == 1:
                            count_mask_graph += 1
                        #mask_graph[global_y,global_x] = 1
                        #plt.plot(global_x,global_y,color=colors[ii%5],marker='+')
                    else:
                        break

                if len(pos_y_vector) > 0:
                    new_segment = []
                    #if visualize_graph:
                        #plt.plot(next_element_x,next_element_y,marker='o',color=colors[count%5])
                    for kk in range(0,len(pos_y_vector)):
                        #mask_graph[pos_y_vector[kk],pos_x_vector[kk]] = 1
                        mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
                        new_segment.append(pos_y_vector[kk]*w+pos_x_vector[kk])
                        segment_ids[pos_y_vector[kk],pos_x_vector[kk]] = count + 1
                    #if visualize_graph:
                        #plt.scatter(pos_x_vector,pos_y_vector,color=colors[count%5],marker='+')
                    count += 1
                    segments.append(new_segment)

                if visualize_graph:
                    plt.imshow(img)
                    indxs = np.argwhere(mask_graph==1)
                    plt.scatter(indxs[:,1],indxs[:,0],color='green',marker='+')
                    plt.show()



            confidence_pending_connections = np.delete(confidence_pending_connections,max_idx)
            pending_connections_x = np.delete(pending_connections_x,max_idx)
            pending_connections_y = np.delete(pending_connections_y,max_idx)

            parent_connections_x = np.delete(parent_connections_x,max_idx)
            parent_connections_y = np.delete(parent_connections_y,max_idx)


            #confident_connections = get_most_confident_outputs(img_idx, next_element_y, next_element_x, confident_th, gpu_id)
            confident_connections = get_most_confident_outputs_vessel_width(img_idx, next_element_y, next_element_x, confident_th, gpu_id)


            #for kk in range(0,len(confident_connections)):
            for kk in range(0,len(confident_connections['peak_value'])):
                tmp_x = confident_connections['x_peak'][kk]+next_element_x-32
                tmp_y = confident_connections['y_peak'][kk]+next_element_y-32
                dist = (tmp_x-center_optical_disk_col)*(tmp_x-center_optical_disk_col) + (tmp_y-center_optical_disk_row)*(tmp_y-center_optical_disk_row)
                #if dist > radius_optical_disk*radius_optical_disk and mask[tmp_y,tmp_x] == 0:
                if mask[tmp_y,tmp_x] == 0:
                #if True:

                    pending_connections_x = np.append(pending_connections_x,tmp_x)
                    pending_connections_y = np.append(pending_connections_y,tmp_y)
                    confidence_pending_connections = np.append(confidence_pending_connections,confident_connections['peak_value'][kk])

                    parent_connections_x = np.append(parent_connections_x,next_element_x)
                    parent_connections_y = np.append(parent_connections_y,next_element_y)

                    min_y = np.max([0, tmp_y-offset])
                    min_x = np.max([0, tmp_x-offset])
                    max_y = np.min([h-1, tmp_y+offset+1])
                    max_x = np.min([w-1, tmp_x+offset+1])
                    mask[min_y:max_y,min_x:max_x] = 1
                    #mask[tmp_y-2:tmp_y+3,tmp_x-2:tmp_x+3] = 1
            ii += 1



        #scipy.misc.imsave('/scratch_net/boxy/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/pred_graph_%02d.png' % img_idx, mask)
        #scipy.misc.imsave('/scratch_net/boxy/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/pred_graph_%02d_mask_graph.png' % img_idx, mask_graph)
        #scipy.misc.imsave('/scratch_net/boxy/carlesv/results/DRIVE/Results_iterative_graph_creation_th_25/pred_graph_%02d.png' % img_idx, final_mask_graph)

        scipy.misc.imsave('/scratch_net/boxy/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/pred_graph_%02d_mask_graph_min_confidence_sp.png' % img_idx, mask_graph)










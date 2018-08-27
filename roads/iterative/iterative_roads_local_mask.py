__author__ = 'carlesv'
from PIL import Image
import torch
from torch.autograd import Variable
import Nets as nt
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np
import roads.patch.bifurcations_toolbox_roads as tb
from astropy.table import Table
import networkx as nx
from scipy import ndimage
import scipy.misc
import random
from skimage.morphology import skeletonize
import matplotlib.patches as patches
import os


def generate_graph_center_roads(img_filename, center, gt_masks, vgg):

    if gt_masks:
        results_dir_vessels = './gt_dbs/MassachusettsRoads/test/1st_manual/'
        pred = Image.open(results_dir_vessels + img_filename[0:len(img_filename)-1])
    else:
        if vgg:
            results_dir_vessels = './results_test_vgg/'
            pred = Image.open(results_dir_vessels + img_filename[:-4] + 'png')
        else:
            results_dir_vessels = './results_test_resnet/'
            pred = Image.open(results_dir_vessels + img_filename)

    pred = np.array(pred)

    patch_size = 64
    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                cost = 255 - pred[row_idx-1,col_idx-1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                cost = 255 - pred[row_idx-1,col_idx]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                cost = 255 - pred[row_idx-1,col_idx+1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                cost = 255 - pred[row_idx,col_idx-1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                cost = 255 - pred[row_idx,col_idx+1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_right_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx > 0:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                cost = 255 - pred[row_idx+1,col_idx-1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                cost = 255 - pred[row_idx+1,col_idx]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx < pred.shape[1]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                cost = 255 - pred[row_idx+1,col_idx+1]
                if cost > 155:
                    cost = cost*10
                G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G


def get_most_confident_outputs(img_filename, patch_center_row, patch_center_col, confident_th, gpu_id):

    patch_size = 64
    center = (patch_center_col, patch_center_row)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    confident_connections = {}
    confident_connections['x_peak'] = []
    confident_connections['y_peak'] = []
    confident_connections['peak_value'] = []

    root_dir='./gt_dbs/MassachusettsRoads/test/images/'
    img = Image.open(os.path.join(root_dir, img_filename))
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
        model_dir = './results_dir/'
        modelName = tb.construct_name(p, "HourGlass")
        numHGScales = 4  # How many times to downsample inside each HourGlass
        net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
        epoch = 130
        net.load_state_dict(torch.load(os.path.join(model_dir, os.path.join(model_dir, modelName+'_epoch-'+str(epoch)+'.pth')),
                                   map_location=lambda storage, loc: storage))

        if gpu_id >= 0:
            net = net.cuda()

        output = net.forward(inputs)
        pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))


        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)

        if visualize_graph_step_by_step:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(img.astype(np.uint8))
            mask_graph_skel = skeletonize(mask_graph>0)
            indxs = np.argwhere(mask_graph_skel==1)
            axes[0].scatter(indxs[:,1],indxs[:,0],color='red',marker='+')

            axes[0].add_patch(patches.Rectangle((x_tmp, y_tmp),patch_size,patch_size,fill=False,color='cyan', linewidth=5))
            img_crop_array = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]
            axes[1].imshow(img_crop_array.astype(np.uint8), interpolation='nearest')
            tmp_vector_x = []
            tmp_vector_y = []
            for ii in range(0,len(sources['peak_value'])):
                if sources['peak_value'][ii] > confident_th:
                    tmp_vector_x.append(sources['x_peak'][ii])
                    tmp_vector_y.append(sources['y_peak'][ii])
            axes[1].plot(tmp_vector_x, tmp_vector_y, ls='none', color='red',marker='+', ms=25, markeredgewidth=10)
            axes[1].plot(32, 32, ls='none', color='cyan',marker='+', ms=25, markeredgewidth=10)
            plt.show()

        if visualize_evolution:

            if iter < 20 or (iter < 200 and iter % 20 == 0) or iter % 100 == 0:

                plt.figure(figsize=(12,12), dpi=60)
                plt.imshow(img.astype(np.uint8))
                mask_graph_skeleton = skeletonize(mask_graph>0)
                indxs_skel = np.argwhere(mask_graph_skeleton==1)
                plt.scatter(indxs_skel[:,1],indxs_skel[:,0],color='red',marker='+')
                plt.axis('off')
                plt.savefig(directory + 'iter_%05d.png' % iter, bbox_inches='tight')
                plt.close()

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



visualize_graph = False
visualize_graph_step_by_step = False
visualize_evolution = False
visualize_dense_evolution = False
save_results = False
gt_masks = False
vgg = True

if visualize_graph or visualize_graph_step_by_step or visualize_evolution or visualize_dense_evolution:
    gpu_id = -1
else:
    gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
    
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)

root_dir='./gt_dbs/MassachusettsRoads/test/images/'
test_img_filenames = os.listdir(root_dir)


for img_idx in range(0,len(test_img_filenames)):

    img_filename = test_img_filenames[img_idx]

    if visualize_evolution:
        directory = './results_dir/results_evolution/' + img_filename[0:len(img_filename)-5] + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)

    if visualize_dense_evolution:
        directory_dense = './results_dir/results_evolution/' + img_filename[0:len(img_filename)-5] + '/dense/'
        if not os.path.exists(directory_dense):
            os.makedirs(directory_dense)

    if gt_masks:
        results_dir_vessels = './gt_dbs/MassachusettsRoads/test/1st_manual/'
        pred = Image.open(results_dir_vessels + img_filename[0:len(img_filename)-1])
        pred = np.array(pred)
        indxs = np.argwhere(pred==255)
        selected_indx = random.randint(0,len(indxs)-1)
        start_row = indxs[selected_indx,0]
        start_col = indxs[selected_indx,1]
    else:
        if vgg:
            results_dir_vessels = './results_test_vgg/'
            pred = Image.open(results_dir_vessels + img_filename[:-4] + 'png')
        else:
            results_dir_vessels = './results_test_resnet/'
            pred = Image.open(results_dir_vessels + img_filename)

        pred = np.array(pred)
        indx_max =  np.argmax(pred)
        start_row = indx_max/pred.shape[1]
        start_col = indx_max%pred.shape[1]



    img = Image.open(os.path.join(root_dir, img_filename))
    if visualize_graph_step_by_step or visualize_graph:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()

    if visualize_evolution or visualize_dense_evolution:
        import matplotlib.pyplot as plt

    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]
    mask_graph = np.zeros((h,w))
    mask_starting_points = np.ones(pred.shape)

    parent_map = {}
    location_map = {}
    current_id = 0
    iter = 0

    exploring = True
    while exploring:

        center_points = []
        center = (start_col, start_row)
        center_points.append(center)
        confident_th = 20
        high_conf_th = 40
        low_conf_th = 15

        colors = ['red', 'blue', 'cyan', 'green', 'purple']

        pending_connections_x = []
        pending_connections_y = []
        parent_connections_x = []
        parent_connections_y = []

        confidence_pending_connections = []
        pending_connections_x.append(center[0])
        pending_connections_y.append(center[1])
        confidence_pending_connections.append(255)

        parent_connections_x.append(center[0])
        parent_connections_y.append(center[1])

        connections_to_be_extended = []
        connections_to_be_extended.append(True)

        parent_map[current_id] = -1
        location_map[current_id] = center

        offset = 2
        offset_local_mask = 10

        ii = 0
        while len(pending_connections_x) > 0:
            max_idx = np.argmax(confidence_pending_connections)
            next_element_x = pending_connections_x[max_idx]
            next_element_y = pending_connections_y[max_idx]

            if ii > 0:

                previous_center_x = parent_connections_x[max_idx]
                previous_center_y = parent_connections_y[max_idx]

                tmp_center = (previous_center_x,previous_center_y)
                G = generate_graph_center_roads(test_img_filenames[img_idx],tmp_center, gt_masks, vgg)

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
                    pos_y_vector.append(global_y)
                    pos_x_vector.append(global_x)


                if len(pos_y_vector) > 0:
                    for kk in range(0,len(pos_y_vector)):
                        mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1

                if visualize_dense_evolution:

                    indxs_mask_graph = np.argwhere(mask_graph>0)
                    tmp_image = np.zeros(img_array.shape)
                    np.copyto(tmp_image, img)
                    tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],0] = 255
                    tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],1] = 0
                    tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],2] = 0
                    scipy.misc.imsave(directory_dense + 'iter_%05d.png' % iter, tmp_image)

                    iter = iter + 1

            confidence_pending_connections = np.delete(confidence_pending_connections,max_idx)
            pending_connections_x = np.delete(pending_connections_x,max_idx)
            pending_connections_y = np.delete(pending_connections_y,max_idx)

            parent_connections_x = np.delete(parent_connections_x,max_idx)
            parent_connections_y = np.delete(parent_connections_y,max_idx)

            parent_id = current_id

            to_be_extended = connections_to_be_extended[max_idx]
            connections_to_be_extended = np.delete(connections_to_be_extended,max_idx)

            if to_be_extended:

                confident_connections = get_most_confident_outputs(test_img_filenames[img_idx], next_element_y, next_element_x, confident_th, gpu_id)

                for kk in range(0,len(confident_connections['peak_value'])):
                    tmp_x = confident_connections['x_peak'][kk]+next_element_x-32
                    tmp_y = confident_connections['y_peak'][kk]+next_element_y-32

                    local_mask = np.zeros((h,w))
                    if parent_id in parent_map.keys():
                        grandparent_id = parent_map[parent_id]
                        if grandparent_id != -1:
                            location_grandparent = location_map[grandparent_id]
                            min_y = np.max([0, location_grandparent[1]-offset_local_mask])
                            min_x = np.max([0, location_grandparent[0]-offset_local_mask])
                            max_y = np.min([h-1, location_grandparent[1]+offset_local_mask+1])
                            max_x = np.min([w-1, location_grandparent[0]+offset_local_mask+1])
                            local_mask[min_y:max_y,min_x:max_x] = 1

                    if local_mask[tmp_y,tmp_x] == 0 and mask_graph[tmp_y,tmp_x] == 0:

                        pending_connections_x = np.append(pending_connections_x,tmp_x)
                        pending_connections_y = np.append(pending_connections_y,tmp_y)
                        confidence_pending_connections = np.append(confidence_pending_connections,confident_connections['peak_value'][kk])

                        parent_connections_x = np.append(parent_connections_x,next_element_x)
                        parent_connections_y = np.append(parent_connections_y,next_element_y)

                        current_id += 1
                        parent_map[current_id] = parent_id
                        location_map[current_id] = (tmp_x, tmp_y)

                        connections_to_be_extended = np.append(connections_to_be_extended,True)

                    elif local_mask[tmp_y,tmp_x] == 0 and mask_graph[tmp_y,tmp_x] == 1:

                        pending_connections_x = np.append(pending_connections_x,tmp_x)
                        pending_connections_y = np.append(pending_connections_y,tmp_y)
                        confidence_pending_connections = np.append(confidence_pending_connections,confident_connections['peak_value'][kk])

                        parent_connections_x = np.append(parent_connections_x,next_element_x)
                        parent_connections_y = np.append(parent_connections_y,next_element_y)

                        current_id += 1
                        parent_map[current_id] = parent_id
                        location_map[current_id] = (tmp_x, tmp_y)

                        connections_to_be_extended = np.append(connections_to_be_extended,False)

            ii += 1


        if visualize_evolution:

            plt.figure(figsize=(12,12), dpi=60)
            plt.imshow(img)
            mask_graph_skeleton = skeletonize(mask_graph>0)
            indxs_skel = np.argwhere(mask_graph_skeleton==1)
            plt.scatter(indxs_skel[:,1],indxs_skel[:,0],color='red',marker='+')
            plt.axis('off')
            plt.savefig(directory + 'iter_%05d.png' % iter, bbox_inches='tight')
            plt.close()

        #See if there are new starting points to explore
        
        mask_detected_vessels = np.ones(pred.shape)
        indxs_vessels = np.argwhere(mask_graph==1)
        mask_detected_vessels[indxs_vessels[:,0],indxs_vessels[:,1]] = 0
        dist_from_roads_detected = ndimage.distance_transform_edt(mask_detected_vessels)

        mask_starting_points[start_row, start_col] = 0
        dist_from_starting_points = ndimage.distance_transform_edt(mask_starting_points)

        dist_from_roads_detected_and_starting_points = (dist_from_roads_detected > 100) * (dist_from_starting_points > 100)


        if gt_masks:
            results_dir_vessels = './gt_dbs/MassachusettsRoads/test/1st_manual/'
            pred = Image.open(results_dir_vessels + img_filename[0:len(img_filename)-1])
            pred = np.array(pred)

            tmp_matrix = (dist_from_roads_detected_and_starting_points == True) * (pred==255)
            indxs = np.argwhere(tmp_matrix==True)
            if len(indxs) > 0:
                selected_indx = random.randint(0,len(indxs)-1)
                start_row = indxs[selected_indx,0]
                start_col = indxs[selected_indx,1]
            else:
                exploring = False
        else:
            if vgg:
                results_dir_vessels = './results_test_vgg/'
                pred = Image.open(results_dir_vessels + img_filename[:-4] + 'png')
            else:
                results_dir_vessels = '/results_test_resnet/'
                pred = Image.open(results_dir_vessels + img_filename)

            pred = np.array(pred)

            pred_to_explore = np.zeros(pred.shape)
            indxs = np.argwhere(dist_from_roads_detected_and_starting_points==True)
            pred_to_explore[indxs[:,0],indxs[:,1]] = pred[indxs[:,0],indxs[:,1]]

            indx_max =  np.argmax(pred_to_explore)
            start_row = indx_max/pred.shape[1]
            start_col = indx_max%pred.shape[1]
            max_val = pred_to_explore[start_row,start_col]
            if max_val < 200:
                exploring = False


    if visualize_graph:
        plt.imshow(img)
        indxs = np.argwhere(mask_graph==1)
        plt.scatter(indxs[:,1],indxs[:,0],color='red',marker='+')
        plt.show()

    if visualize_evolution:

        plt.figure(figsize=(12,12), dpi=60)
        plt.imshow(img)
        mask_graph_skeleton = skeletonize(mask_graph>0)
        indxs_skel = np.argwhere(mask_graph_skeleton==1)
        plt.scatter(indxs_skel[:,1],indxs_skel[:,0],color='red',marker='+')
        plt.axis('off')
        plt.savefig(directory + 'iter_%05d.png' % iter, bbox_inches='tight')
        plt.close()

    if visualize_dense_evolution:
        os.system('ffmpeg -framerate 10 -i ' + directory_dense + '/iter_%05d.png -c:v libx264 -vf "fps=10,format=yuv420p" ' + directory_dense + 'output.mp4')
        os.system('rm '+ directory_dense + '*.png')
        indxs_mask_graph = np.argwhere(mask_graph>0)
        tmp_image = np.zeros(img_array.shape)
        np.copyto(tmp_image, img)
        tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],0] = 255
        tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],1] = 0
        tmp_image[indxs_mask_graph[:,0],indxs_mask_graph[:,1],2] = 0
        scipy.misc.imsave(directory_dense + 'last_iter_img.png', tmp_image)
        scipy.misc.imsave(directory_dense + 'last_iter_mask.png', mask_graph)

    if save_results:
        if gt_masks:
            scipy.misc.imsave('./results_dir/iterative_results_ground_truth/' + img_filename, mask_graph)
        else:
            if vgg:
                scipy.misc.imsave('./results_dir/iterative_results_prediction_vgg/' + img_filename, mask_graph)
            else:
                scipy.misc.imsave('/results_dir/iterative_results_prediction_resnet/' + img_filename, mask_graph)


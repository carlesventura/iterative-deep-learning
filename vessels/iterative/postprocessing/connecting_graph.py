__author__ = 'carlesv'

from PIL import Image
import numpy as np
import cv2
import networkx as nx
from scipy import ndimage
import vessels.iterative.shortest_path as sp
import scipy.misc

def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)

    #out = np.zeros_like(skel)
    #out[np.where(filtered==11)] = 1
    #return out

    return np.where(filtered==11)

def find_connected_points(skel, point, connected_points, depth):

    h, w = skel.shape[:2]
    point_row = point / w
    point_col = point % w

    neigh_row = point_row-1
    neigh_col = point_col-1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row-1
    neigh_col = point_col
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row-1
    neigh_col = point_col+1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row
    neigh_col = point_col-1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row
    neigh_col = point_col+1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row+1
    neigh_col = point_col-1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row+1
    neigh_col = point_col
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)

    neigh_row = point_row+1
    neigh_col = point_col+1
    neigh_point = neigh_row*w + neigh_col
    if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
        connected_points.append(neigh_point)
        find_connected_points(skel, neigh_point, connected_points, depth+1)


min_confidence = True
confidence_th = 40
skeletonized = True

#results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_th_25/'
#results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation/'
results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/'

for img_idx in range(1,21):

    #skel = Image.open(results_dir + 'pred_graph_%02d.png' %(img_idx))
    skel = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_040_skeleton.png' %(img_idx))
    skel = np.array(skel)
    h, w = skel.shape[:2]
    if not skeletonized:
        #skel = skeletonize(skel==255)
        #scipy.misc.imsave(results_dir + 'pred_graph_%02d_skeleton.png' % img_idx, skel)
        skel = Image.open(results_dir + 'pred_graph_%02d_skeleton.png' %(img_idx))
        skel = np.array(skel)
        #print(np.max(skel))
    skel_endpoints = skeleton_endpoints(skel)

    for ii in range(0,len(skel_endpoints[0])):
        endpoint_row = skel_endpoints[0][ii]
        endpoint_col = skel_endpoints[1][ii]
        mask_endpoint = np.ones((h,w))
        mask_endpoint[endpoint_row,endpoint_col] = 0
        dist_endpoint = ndimage.distance_transform_edt(mask_endpoint)

        #Set high distance to elements not belonging to the skeleton
        indxs = np.argwhere(skel==0)
        dist_endpoint[indxs[:,0],indxs[:,1]] = 1000

        #Set high distance to elements belonging to the same connected part of the skeleton as the endpoint
        endpoint = endpoint_row*w + endpoint_col
        connected_points = [endpoint]
        find_connected_points(skel, endpoint, connected_points, 0)
        for jj in range(0,len(connected_points)):
            pos_row = connected_points[jj]/w
            pos_col = connected_points[jj]%w
            dist_endpoint[pos_row,pos_col] = 1000

        min_idx = np.argmin(dist_endpoint)
        min_val = np.min(dist_endpoint)


        # print(min_val)
        # plt.imshow(skel)
        # plt.scatter(min_idx%w,min_idx/w,color='red')
        # plt.scatter(endpoint_col,endpoint_row,color='green')
        # plt.show()


        if min_val < 10:
            tmp_center = (endpoint_col,endpoint_row)
            patch_size = 24

            if endpoint_col < patch_size/2:
                patch_size = 2*endpoint_col
            if endpoint_col > w - patch_size/2:
                patch_size = 2*(w-endpoint_col)
            if endpoint_row < patch_size/2:
                patch_size = 2*endpoint_row
            if endpoint_row > h - patch_size/2:
                patch_size = 2*(h-endpoint_row)

            if min_confidence:
                G = sp.generate_graph_center_patch_size_min_confidence(img_idx,tmp_center,patch_size, confidence_th)
            else:
                G = sp.generate_graph_center_patch_size(img_idx,tmp_center,patch_size)
            target_idx = (patch_size/2)*patch_size + (patch_size/2)
            row_pos = min_idx / w
            col_pos = min_idx % w
            source_idx = (row_pos-endpoint_row+(patch_size/2))*patch_size + col_pos-endpoint_col+(patch_size/2)
            length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)

            #print(avg_cost)
            #print(len(path))
            #plt.imshow(skel)
            #plt.scatter(min_idx%w,min_idx/w,color='green')
            #plt.scatter(endpoint_col,endpoint_row,color='green')

            if min_confidence:
                if length < 1e8 and len(path) < 15:
                    for jj in range(1,len(path)-1):
                        node_idx = path[jj]
                        row_idx = node_idx / patch_size + endpoint_row-(patch_size/2)
                        col_idx = node_idx % patch_size + endpoint_col-(patch_size/2)
                        #plt.scatter(col_idx,row_idx,color='red')
                        skel[row_idx,col_idx] = 255
            else:
                avg_cost = float(length) / len(path)
                if avg_cost < 100 and len(path) < 15:
                    for jj in range(1,len(path)-1):
                        node_idx = path[jj]
                        row_idx = node_idx / patch_size + endpoint_row-(patch_size/2)
                        col_idx = node_idx % patch_size + endpoint_col-(patch_size/2)
                        #plt.scatter(col_idx,row_idx,color='red')
                        skel[row_idx,col_idx] = 255

            #plt.show()

    if min_confidence:
        #scipy.misc.imsave(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_%03d.png' % (img_idx, confidence_th), skel)
        scipy.misc.imsave(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_%03d_2n.png' % (img_idx, confidence_th), skel)
    else:
        scipy.misc.imsave(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20.png' % img_idx, skel)

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


def find_connected_points_until_junction(skel, point, connected_points, depth):

    h, w = skel.shape[:2]
    point_row = point / w
    point_col = point % w

    junction_found = False
    count = 0
    if skel[point_row-1,point_col-1] == 255:
        count += 1
    if skel[point_row-1,point_col] == 255:
        count += 1
    if skel[point_row-1,point_col+1] == 255:
        count += 1
    if skel[point_row,point_col-1] == 255:
        count += 1
    if skel[point_row,point_col+1] == 255:
        count += 1
    if skel[point_row+1,point_col-1] == 255:
        count += 1
    if skel[point_row+1,point_col] == 255:
        count += 1
    if skel[point_row+1,point_col+1] == 255:
        count += 1

    if count > 2:
        junction_found = True

    if not junction_found:
        neigh_row = point_row-1
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row-1
        neigh_col = point_col
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row-1
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row+1
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row+1
        neigh_col = point_col
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)

        neigh_row = point_row+1
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points_until_junction(skel, neigh_point, connected_points, depth+1)


confidence_th = 100

results_dir = './results_dir_vessels/results_iterative_graph_creation_no_mask_offset_th_25/'

for img_idx in range(1,21):

    skel = Image.open(results_dir + 'pred_graph_%02d_mask_graph_offset_2_dilated_skeleton.png' %(img_idx))
    skel = np.array(skel)
    h, w = skel.shape[:2]
    skel_endpoints = skeleton_endpoints(skel)

    for ii in range(0,len(skel_endpoints[0])):

        endpoint_row = skel_endpoints[0][ii]
        endpoint_col = skel_endpoints[1][ii]
        mask_endpoint = np.ones((h,w))
        mask_endpoint[endpoint_row,endpoint_col] = 0
        dist_endpoint = ndimage.distance_transform_edt(mask_endpoint)

        #Set high distance to elements belonging to the skeleton
        indxs = np.argwhere(skel>0)
        dist_endpoint[indxs[:,0],indxs[:,1]] = 1000

        #Set high distance to elements belonging to the same connected part of the skeleton as the endpoint
        endpoint = endpoint_row*w + endpoint_col
        connected_points = [endpoint]
        find_connected_points_until_junction(skel, endpoint, connected_points, 0)
        diff_x = []
        diff_y = []
        for jj in range(1,len(connected_points)):
            pos_row = connected_points[jj]/w
            pos_col = connected_points[jj]%w
            diff_x.append(pos_col-endpoint_col)
            diff_y.append(pos_row-endpoint_row)

        mean_diff_x = np.mean(diff_x)
        mean_diff_y = np.mean(diff_y)
        norm_diff = np.sqrt(mean_diff_x*mean_diff_x+mean_diff_y*mean_diff_y)

        mean_diff_x = mean_diff_x / norm_diff
        mean_diff_y = mean_diff_y / norm_diff

        endbranch_found = False
        dist_low_th = 0
        dist_high_th = 2
        best_candidate = endpoint
        best_candidate_row = endpoint_row
        best_candidate_col = endpoint_col
        while not endbranch_found and dist_high_th < 20:
            indxs_dist = (dist_endpoint>dist_low_th)*(dist_endpoint<=dist_high_th)
            indxs_dist = np.where(indxs_dist)
            candidates = []
            cost_candidates = []
            for jj in range(0,len(indxs_dist[0])):
                diff_x = indxs_dist[1][jj] - endpoint_col
                diff_y = indxs_dist[0][jj] - endpoint_row
                norm_diff = np.sqrt(diff_x*diff_x+diff_y*diff_y)
                diff_x = diff_x / norm_diff
                diff_y = diff_y / norm_diff
                cos_angle = diff_x*mean_diff_x + diff_y*mean_diff_y
                if cos_angle < 0:

                    tmp_center = (endpoint_col,endpoint_row)
                    patch_size = 2*dist_high_th+2
                    if endpoint_row - (patch_size/2) > 0 and endpoint_row + (patch_size/2) < h and endpoint_col - (patch_size/2) > 0 and endpoint_col + (patch_size/2) < w:
                        candidates.append(jj)
                        G = sp.generate_graph_center_patch_size_min_confidence(img_idx,tmp_center,patch_size, confidence_th)
                        target_idx = (patch_size/2)*patch_size + (patch_size/2)
                        row_pos = indxs_dist[0][jj]
                        col_pos = indxs_dist[1][jj]
                        source_idx = (row_pos-endpoint_row+(patch_size/2))*patch_size + col_pos-endpoint_col+(patch_size/2)
                        length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
                        cost_candidates.append(length)

            if len(candidates) > 0:

                min_cost_candidate = np.min(cost_candidates)
                min_cost_candidate_indx = np.argmin(cost_candidates)
                if min_cost_candidate > 1e7:
                    endbranch_found = True
                else:
                    best_candidate_row = indxs_dist[0][candidates[min_cost_candidate_indx]]
                    best_candidate_col = indxs_dist[1][candidates[min_cost_candidate_indx]]
                    best_candidate = best_candidate_row*w + best_candidate_col
                    dist_low_th += 2
                    dist_high_th += 2

            else:

                endbranch_found = True

        dist_high_th -= 2
        if best_candidate != endpoint:
            tmp_center = (endpoint_col,endpoint_row)
            patch_size = 2*dist_high_th+2
            G = sp.generate_graph_center_patch_size_min_confidence(img_idx,tmp_center,patch_size, confidence_th)
            target_idx = (patch_size/2)*patch_size + (patch_size/2)
            row_pos = best_candidate_row
            col_pos = best_candidate_col
            source_idx = (row_pos-endpoint_row+(patch_size/2))*patch_size + col_pos-endpoint_col+(patch_size/2)
            length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)
            for jj in range(0,len(path)):
                node_idx = path[jj]
                row_idx = node_idx / patch_size + endpoint_row-(patch_size/2)
                col_idx = node_idx % patch_size + endpoint_col-(patch_size/2)
                skel[row_idx,col_idx] = 255


    scipy.misc.imsave(results_dir + 'pred_graph_%02d_mask_graph_extended_branches_confidence_th_%03d.png' % (img_idx, confidence_th), skel)



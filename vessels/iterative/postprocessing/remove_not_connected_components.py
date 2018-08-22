__author__ = 'carlesv'

from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
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

def find_connected_points(skel, point, connected_points, depth, max_reached_depth):

    if depth > max_reached_depth[0]:
        max_reached_depth[0] = depth

    h, w = skel.shape[:2]
    point_row = point / w
    point_col = point % w


    neigh_row = point_row-1
    neigh_col = point_col-1
    if neigh_row >= 0 and neigh_col >= 0:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row-1
    neigh_col = point_col
    if neigh_row >= 0:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row-1
    neigh_col = point_col+1
    if neigh_row >= 0 and neigh_col < w:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row
    neigh_col = point_col-1
    if neigh_col >= 0:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row
    neigh_col = point_col+1
    if neigh_col < w:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row+1
    neigh_col = point_col-1
    if neigh_row < h and neigh_col >=0:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row+1
    neigh_col = point_col
    if neigh_row < h:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)

    neigh_row = point_row+1
    neigh_col = point_col+1
    if neigh_row < h and neigh_col < w:
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and depth < 20:
            connected_points.append(neigh_point)
            find_connected_points(skel, neigh_point, connected_points, depth+1, max_reached_depth)



skeletonized = True

#results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_th_25/'
#results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation/'
#results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/'
results_dir = '/scratch/carlesv/results/DRIVE/Results_DRIU_point_supervision_skeleton/pred_th_100/'

for img_idx in range(1,21):

    skel = Image.open(results_dir + 'pred_graph_%02d.png' %(img_idx))
    #skel = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_100_2nd_step_extended_branches_skeleton.png' %(img_idx))
    skel = np.array(skel)
    h, w = skel.shape[:2]

    if not skeletonized:
        #skel = skeletonize(skel==255)
        #scipy.misc.imsave(results_dir + 'pred_graph_%02d_skeleton.png' % img_idx, skel)
        skel = Image.open(results_dir + 'pred_graph_%02d_skeleton.png' %(img_idx))
        skel = np.array(skel)
        #print(np.max(skel))
    skel_endpoints = skeleton_endpoints(skel)

    #Removing isolated points (they are not detected as endpoints)
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel/255,src_depth,kernel)
    skel[np.where(filtered==10)] = 0


    for ii in range(0,len(skel_endpoints[0])):
        #print(str(ii+1) + " / " + str(len(skel_endpoints[0])))
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
        max_reached_depth = [0]
        find_connected_points(skel, endpoint, connected_points, 0, max_reached_depth)
        #print(max_reached_depth[0])

        if max_reached_depth[0] < 20:
            for jj in range(0,len(connected_points)):
                pos_row = connected_points[jj]/w
                pos_col = connected_points[jj]%w
                skel[pos_row, pos_col] = 0
                #if len(connected_points) == 1:
                    #print(np.array([pos_row, pos_col]))
                    #plt.scatter(pos_col,pos_row,color='red')


    #scipy.misc.imsave(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_100_2nd_step_extended_branches_wo_not_connected.png' % (img_idx), skel)
    scipy.misc.imsave(results_dir + 'pred_graph_%02d_wo_not_connected.png' % (img_idx), skel)



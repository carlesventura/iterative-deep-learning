__author__ = 'carlesv'

from PIL import Image
import numpy as np
import os
import networkx as nx
from scipy import ndimage
import cv2

def build_graph(pred):

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                if pred[row_idx-1,col_idx-1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                if pred[row_idx-1,col_idx]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                if pred[row_idx-1,col_idx+1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                if pred[row_idx,col_idx-1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                if pred[row_idx,col_idx+1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_right_idx,weight=cost)

            if col_idx > 0 and row_idx < pred.shape[0]-1:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                if pred[row_idx+1,col_idx-1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                if pred[row_idx+1,col_idx]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if col_idx < pred.shape[1]-1 and row_idx < pred.shape[0]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                if pred[row_idx+1,col_idx+1]:
                    cost = 1
                else:
                    cost = 1e10
                G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

def build_graph_gt(pred):

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                if pred[row_idx-1,col_idx-1]:
                    cost = 1
                    G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                if pred[row_idx-1,col_idx]:
                    cost = 1
                    G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                if pred[row_idx-1,col_idx+1]:
                    cost = 1
                    G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                if pred[row_idx,col_idx-1]:
                    cost = 1
                    G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                if pred[row_idx,col_idx+1]:
                    cost = 1
                    G.add_edge(node_idx,node_right_idx,weight=cost)

            if col_idx > 0 and row_idx < pred.shape[0]-1:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                if pred[row_idx+1,col_idx-1]:
                    cost = 1
                    G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                if pred[row_idx+1,col_idx]:
                    cost = 1
                    G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if col_idx < pred.shape[1]-1 and row_idx < pred.shape[0]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                if pred[row_idx+1,col_idx+1]:
                    cost = 1
                    G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

def evaluate_connectivity(edges_gt, G_gt, pred, G_pred, visualize_connectivity, visualize_errors):

    matching_th = 0.8
    connected = 0

    h, w = pred.shape[:2]

    for ii in range(0,len(edges_gt)):

        mask_start_point = np.ones((h,w))
        mask_end_point = np.ones((h,w))

        edge = edges_gt[ii]
        start_point = edge[0]
        start_point_row = start_point/w
        start_point_col = start_point%w
        mask_start_point[start_point_row,start_point_col] = 0

        end_point = edge[-1]
        end_point_row = end_point/w
        end_point_col = end_point%w
        mask_end_point[end_point_row,end_point_col] = 0

        dist_start_point = ndimage.distance_transform_edt(mask_start_point)
        dist_end_point = ndimage.distance_transform_edt(mask_end_point)

        indxs = np.argwhere(pred==False)

        dist_start_point[indxs[:,0],indxs[:,1]] = 3000
        dist_end_point[indxs[:,0],indxs[:,1]] = 3000
        min_idx_start = np.argmin(dist_start_point)
        min_idx_end = np.argmin(dist_end_point)


        length_pred, path_pred = nx.bidirectional_dijkstra(G_pred, min_idx_start, min_idx_end, weight='weight')
        length_gt, path_gt = nx.bidirectional_dijkstra(G_gt, start_point, end_point, weight='weight')

        connectivity = float(np.min([length_gt,length_pred]))/np.max([length_gt,length_pred])
        if connectivity > matching_th:
            connected += 1
        if connectivity <= matching_th and visualize_errors:

            plt.imshow(pred)
            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path_pred)):
                pos_y = path_pred[jj] / pred.shape[1]
                pos_x = path_pred[jj] % pred.shape[1]
                pos_y_vector.append(pos_y)
                pos_x_vector.append(pos_x)

            plt.scatter(pos_x_vector[0],pos_y_vector[0],color='red',marker='o')
            plt.scatter(pos_x_vector[-1],pos_y_vector[-1],color='red',marker='o')
            plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='+')

            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path_gt)):
                pos_y = path_gt[jj] / pred.shape[1]
                pos_x = path_gt[jj] % pred.shape[1]
                pos_y_vector.append(pos_y)
                pos_x_vector.append(pos_x)

            plt.scatter(pos_x_vector[0],pos_y_vector[0],color='green',marker='o')
            plt.scatter(pos_x_vector[-1],pos_y_vector[-1],color='green',marker='o')
            plt.scatter(pos_x_vector,pos_y_vector,color='green',marker='+')
            plt.show()



        if visualize_connectivity and ii%10 == 0:

            print(length_pred)
            print(length_gt)
            print(float(np.min([length_gt,length_pred]))/np.max([length_gt,length_pred]))

            plt.imshow(pred)
            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path_pred)):
                pos_y = path_pred[jj] / pred.shape[1]
                pos_x = path_pred[jj] % pred.shape[1]
                pos_y_vector.append(pos_y)
                pos_x_vector.append(pos_x)

            plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='o')

            pos_y_vector = []
            pos_x_vector = []
            for jj in range(0,len(path_gt)):
                pos_y = path_gt[jj] / pred.shape[1]
                pos_x = path_gt[jj] % pred.shape[1]
                pos_y_vector.append(pos_y)
                pos_x_vector.append(pos_x)

            plt.scatter(pos_x_vector,pos_y_vector,color='green',marker='+')
            plt.show()

    print('connected = ' + str(connected))
    print('total = ' + str(len(edges_gt)))

    CRR = float(connected) / len(edges_gt)
    return CRR

def find_junctions(graph_gt_img):

    skel = graph_gt_img.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    return np.where(filtered>12)

def find_endpoints(graph_gt_img):

    skel = graph_gt_img.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    return np.where(filtered==11)



def find_connected_points_until_junction_found(skel, point, connected_points, junction_idxs, endpoint_idxs):

    h, w = skel.shape[:2]
    next_point_found = True

    while (point not in junction_idxs) and (point not in endpoint_idxs) and next_point_found:
        point_row = point / w
        point_col = point % w
        next_point_found = False

        neigh_row = point_row-1
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row-1
        neigh_col = point_col
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row-1
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row+1
        neigh_col = point_col-1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row+1
        neigh_col = point_col
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True

        neigh_row = point_row+1
        neigh_col = point_col+1
        neigh_point = neigh_row*w + neigh_col
        if skel[neigh_row,neigh_col] == 255 and neigh_point not in connected_points and not next_point_found:
            connected_points.append(neigh_point)
            point = neigh_point
            next_point_found = True


def extract_edges_from_gt_annotations(graph_gt_img):


    junctions = find_junctions(graph_gt_img)
    endpoints = find_endpoints(graph_gt_img)
    edges_gt = []

    h, w = graph_gt_img.shape[:2]
    junction_idxs = []
    #plt.imshow(graph_gt_img)
    for ii in range(0,len(junctions[0])):
        junction_row = junctions[0][ii]
        junction_col = junctions[1][ii]
        #plt.scatter(junction_col,junction_row,color='green',marker='+')
        junction_idx = junction_row*w + junction_col
        junction_idxs.append(junction_idx)
    endpoint_idxs = []
    for ii in range(0,len(endpoints[0])):
        endpoint_row = endpoints[0][ii]
        endpoint_col = endpoints[1][ii]
        #plt.scatter(endpoint_col,endpoint_row,color='green',marker='o')
        endpoint_idx = endpoint_row*w + endpoint_col
        endpoint_idxs.append(endpoint_idx)
    #plt.show()

    for ii in range(0,len(junctions[0])):
        junction_row = junctions[0][ii]
        junction_col = junctions[1][ii]

        neigh_row = junction_row-1
        neigh_col = junction_col-1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row-1
        neigh_col = junction_col
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row-1
        neigh_col = junction_col+1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row
        neigh_col = junction_col-1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row
        neigh_col = junction_col+1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row+1
        neigh_col = junction_col-1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row+1
        neigh_col = junction_col
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

        neigh_row = junction_row+1
        neigh_col = junction_col+1
        neigh_point = neigh_row*w + neigh_col
        if graph_gt_img[neigh_row,neigh_col] == 255:
            connected_points = []
            connected_points.append(junction_idxs[ii])
            connected_points.append(neigh_point)
            find_connected_points_until_junction_found(graph_gt_img, neigh_point, connected_points, junction_idxs, endpoint_idxs)
            if connected_points[-1] in endpoint_idxs:
                if len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            elif connected_points[-1] in junction_idxs:
                idx = np.argwhere(junction_idxs==connected_points[-1])
                if idx > ii and len(connected_points) > 5:
                    edge = []
                    for jj in range(0,len(connected_points)):
                        edge.append(connected_points[jj])
                    edges_gt.append(edge)
            else:
                print('Error: last point found was not either an endpoint or a junction')

    return edges_gt


visualize_connectivity = False
visualize_errors = False

if visualize_connectivity or visualize_errors:
    import matplotlib.pyplot as plt

iterative_graph_creation = True
vgg = True

if iterative_graph_creation:
    if vgg:
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_th_20_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_width_th_20_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_novelty_th_20_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_min_conf_50_th_20_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_th_30_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_th_20_offset_mask_10_debugging_dilated/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_130_vgg_high_th_40_low_th_15_offset_mask_10/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_130_vgg_high_th_40_low_th_15_offset_mask_10_dilated/'
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_130_vgg_th_20_offset_mask_10/'
        results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_130_vgg_th_30_local_mask_skeleton/'

    else:
        #results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_novelty_dilated/'
        results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_width_offset_mask_10/'
else: #Road segmentation skeleton
    if vgg:
        results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/test_results_vgg_skeletons/th_150/'
    else:
        results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/test_results_road_segmentation_skeletons/th_100/'


CRR_all = []

root_dir='/scratch_net/boxy/carlesv/gt_dbs/MassachusettsRoads/test/images/'
test_img_filenames = os.listdir(root_dir)

if iterative_graph_creation:
    if vgg:
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_vgg_th_20_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_vgg_width_th_20_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_vgg_novelty_th_20_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_vgg_min_conf_50_th_20_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_vgg_th_30_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_vgg_th_20_offset_mask_10_debugging_dilated.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_130_vgg_high_th_40_low_th_15_offset_mask_10.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_130_vgg_high_th_40_low_th_15_offset_mask_10_dilated.txt', 'a')
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_130_vgg_th_20_offset_mask_10.txt', 'a')
        file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_130_vgg_th_30_local_mask.txt', 'a')
    else:
        #file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative.txt', 'a')
        file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_iterative_width_offset_mask_10.txt', 'a')
else:
    if vgg:
        file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_vgg_skeleton_th_150.txt', 'a')
    else:
        file = open('/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/results_connectivity_skeleton_th_100.txt', 'a')

for img_idx in range(0,len(test_img_filenames)):


    img_filename = test_img_filenames[img_idx]
    img = Image.open(os.path.join(root_dir, img_filename))
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    #Predicted graph
    pred = Image.open(results_dir + img_filename)
    pred = np.array(pred)
    G_pred = build_graph(pred)


    print('Predicted graph built')


    #Ground truth graph

    gt_dir='/scratch_net/boxy/carlesv/gt_dbs/MassachusettsRoads/test/1st_manual_skeletons/'
    graph_gt_img = Image.open(gt_dir + img_filename)
    graph_gt_img = np.array(graph_gt_img)
    graph_gt_img[0,:] = 0
    graph_gt_img[:,0] = 0
    graph_gt_img[-1,:] = 0
    graph_gt_img[:,-1] = 0
    G_gt = build_graph_gt(graph_gt_img)
    edges_gt = extract_edges_from_gt_annotations(graph_gt_img)

    print('GT graph built')


    # Find matching between predicted edges and ground truth edges
    CRR = evaluate_connectivity(edges_gt, G_gt, pred, G_pred, visualize_connectivity, visualize_errors)

    print(CRR)
    file.write(img_filename + ': gt edges: ' + str(len(edges_gt)) + '\n')
    file.write(img_filename + ': CRR: %.5f' % (CRR) + '\n')
    file.flush()

    CRR_all.append(CRR)


print(CRR_all)
print(np.mean(CRR_all))

file.write(str(CRR_all) + '\n')
file.write(str(np.mean(CRR_all)) + '\n')
file.flush()
file.close()









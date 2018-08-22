__author__ = 'carlesv'

from PIL import Image
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import scipy.io as sio
from bresenham import bresenham
from scipy import ndimage
import matplotlib.markers as mmarkers

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

def evaluate_connectivity(img_idx, edges_gt, G_gt, pred, G_pred, visualize_connectivity, to_be_cropped):

    matching_th = 0.8
    #matching_th = 0.7
    mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %img_idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    connected = 0
    edges_gt_inside_roi = 0
    for ii in range(0,len(edges_gt)):
        mask_start_point = np.ones((h,w))
        mask_end_point = np.ones((h,w))
        edge = edges_gt[ii]
        inside_roi = True
        for jj in range(0,len(edge)-1):
            src = edge[jj]
            src_x = int(np.round(vertices[src,0]))
            src_y = int(np.round(vertices[src,1]))
            target = edge[jj+1]
            target_x = int(np.round(vertices[target,0]))
            target_y = int(np.round(vertices[target,1]))
            line = list(bresenham(src_x, src_y, target_x, target_y))

            if not to_be_cropped:
                if jj == 0:
                    mask_start_point[line[0][1],line[0][0]] = 0
                    gt_idx_start = line[0][1]*w + line[0][0]
                if jj == len(edge)-2:
                    mask_end_point[line[len(line)-1][1],line[len(line)-1][0]] = 0
                    gt_idx_end = line[len(line)-1][1]*w + line[len(line)-1][0]
            else:
                for kk in range(0,len(line)):
                    if line[kk][0] < 27 or line[kk][0] > 538 or line[kk][1] < 37 or line[kk][1] > 548:
                        inside_roi = False
                #if not inside_roi:
                    #print(np.array([ii, line[0][0], line[0][1] ]))

        if to_be_cropped and inside_roi:

            edges_gt_inside_roi += 1
            for jj in range(0,len(edge)-1):
                src = edge[jj]
                src_x = int(np.round(vertices[src,0]))
                src_y = int(np.round(vertices[src,1]))
                target = edge[jj+1]
                target_x = int(np.round(vertices[target,0]))
                target_y = int(np.round(vertices[target,1]))
                line = list(bresenham(src_x, src_y, target_x, target_y))

                if jj == 0:
                    mask_start_point[line[0][1],line[0][0]] = 0
                    gt_idx_start = line[0][1]*w + line[0][0]
                if jj == len(edge)-2:
                    mask_end_point[line[len(line)-1][1],line[len(line)-1][0]] = 0
                    gt_idx_end = line[len(line)-1][1]*w + line[len(line)-1][0]

        if not to_be_cropped or (to_be_cropped and inside_roi):

            dist_start_point = ndimage.distance_transform_edt(mask_start_point)
            dist_end_point = ndimage.distance_transform_edt(mask_end_point)

            indxs = np.argwhere(pred==False)

            if to_be_cropped:
                indxs = indxs + np.array([37,27])
                dist_start_point[0:37,:] = 1000
                dist_start_point[549:584,:] = 1000
                dist_start_point[:,0:27] = 1000
                dist_start_point[:,539:565] = 1000
                dist_end_point[0:37,:] = 1000
                dist_end_point[549:584,:] = 1000
                dist_end_point[:,0:27] = 1000
                dist_end_point[:,539:565] = 1000

            dist_start_point[indxs[:,0],indxs[:,1]] = 1000
            dist_end_point[indxs[:,0],indxs[:,1]] = 1000
            min_idx_start = np.argmin(dist_start_point)
            min_idx_end = np.argmin(dist_end_point)

            if to_be_cropped:
                min_idx_start_row = min_idx_start / w - 37
                min_idx_start_col = min_idx_start % w - 27
                min_idx_start = min_idx_start_row*512 + min_idx_start_col
                min_idx_end_row = min_idx_end / w - 37
                min_idx_end_col = min_idx_end % w - 27
                min_idx_end = min_idx_end_row*512 + min_idx_end_col

            length_pred, path_pred = nx.bidirectional_dijkstra(G_pred, min_idx_start, min_idx_end, weight='weight')
            length_gt, path_gt = nx.bidirectional_dijkstra(G_gt, gt_idx_start, gt_idx_end, weight='weight')

            connectivity = float(np.min([length_gt,length_pred]))/np.max([length_gt,length_pred])
            if connectivity > matching_th:
                connected += 1
            if connectivity <= matching_th and visualize_errors:
            #if visualize_errors:
                #print(length_pred)
                #print(length_gt)
                #print(float(np.min([length_gt,length_pred]))/np.max([length_gt,length_pred]))

                #fig, axes = plt.subplots(2, 2)
                #axes[0,0].imshow(img)
                #axes[0,1].imshow(pred)

                plt.imshow(pred)
                pos_y_vector = []
                pos_x_vector = []
                for jj in range(0,len(path_pred)):
                    pos_y = path_pred[jj] / pred.shape[1]
                    pos_x = path_pred[jj] % pred.shape[1]
                    pos_y_vector.append(pos_y)
                    pos_x_vector.append(pos_x)

                #plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='o')

                #axes[1,0].imshow(img)
                #axes[1,1].imshow(pred)
                #axes[1,0].scatter(pos_x_vector,pos_y_vector,color='red',marker='o')
                #axes[1,1].scatter(pos_x_vector,pos_y_vector,color='red',marker='o')

                pos_y_vector = []
                pos_x_vector = []
                for jj in range(0,len(path_gt)):
                    pos_y = path_gt[jj] / pred.shape[1]
                    pos_x = path_gt[jj] % pred.shape[1]
                    pos_y_vector.append(pos_y)
                    pos_x_vector.append(pos_x)

                plt.scatter(pos_x_vector[0],pos_y_vector[0],color='red',marker='o')
                plt.scatter(pos_x_vector[-1],pos_y_vector[-1],color='red',marker='o')
                plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='+')
                #axes[1,0].scatter(pos_x_vector,pos_y_vector,color='green',marker='+')
                #axes[1,1].scatter(pos_x_vector,pos_y_vector,color='green',marker='+')
                #plt.show()



            if visualize_connectivity and ii%10 == 0:

                print(length_pred)
                print(length_gt)
                print(float(np.min([length_gt,length_pred]))/np.max([length_gt,length_pred]))

                #plt.imshow(img)
                #plt.imshow(pred)

                pred_img = np.zeros((584,565,3),dtype=np.uint8)
                no_pred_indxs = np.argwhere(pred==0)
                pred_indxs = np.argwhere(pred>0)
                pred_img[no_pred_indxs[:,0],no_pred_indxs[:,1],:]=255
                pred_img[pred_indxs[:,0],pred_indxs[:,1],1:]=255
                plt.imshow(pred_img)

                pos_y_vector = []
                pos_x_vector = []
                for jj in range(0,len(path_gt)):
                    pos_y = path_gt[jj] / pred.shape[1]
                    pos_x = path_gt[jj] % pred.shape[1]
                    pos_y_vector.append(pos_y)
                    pos_x_vector.append(pos_x)

                marker = mmarkers.MarkerStyle(marker='o', fillstyle='none')
                plt.scatter(pos_x_vector,pos_y_vector, color='green', facecolors='none',s=100,linewidth=2)
                pos_y_vector = []
                pos_x_vector = []
                for jj in range(0,len(path_pred)):
                    pos_y = path_pred[jj] / pred.shape[1]
                    pos_x = path_pred[jj] % pred.shape[1]
                    pos_y_vector.append(pos_y)
                    pos_x_vector.append(pos_x)

                #plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='o')
                plt.scatter(pos_x_vector,pos_y_vector,color='red',marker='+',s=100,linewidth=2)

                plt.show()

    plt.show()

    if to_be_cropped:
        print(edges_gt_inside_roi)
        print(len(edges_gt))
        CRR = float(connected) / edges_gt_inside_roi
    else:
        CRR = float(connected) / len(edges_gt)
    return CRR


def extract_edges_from_gt_annotations(img_idx):

    mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %img_idx)
    subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])-1
    junctions = np.squeeze(mat_contents['G']['junctions'][0,0])-1
    edges_gt = []


    for ii in range(0,len(junctions)):
        next_idxs = np.argwhere(subscripts==junctions[ii])
        num_edges = len(next_idxs)
        for jj in range(0,num_edges):
            edge = [junctions[ii]]
            if next_idxs[jj,1] == 0:
                next_vertex = subscripts[next_idxs[jj,0],1]
            else:
                next_vertex = subscripts[next_idxs[jj,0],0]
            previous_vertex = junctions[ii]
            while next_vertex not in junctions:
                edge.append(next_vertex)
                next_idxs_tmp = np.argwhere(subscripts==next_vertex)
                if next_idxs_tmp[0,1] == 0:
                    next_vertex_tmp = subscripts[next_idxs_tmp[0,0],1]
                else:
                    next_vertex_tmp = subscripts[next_idxs_tmp[0,0],0]
                if next_vertex_tmp != previous_vertex:
                    previous_vertex = next_vertex
                    next_vertex = next_vertex_tmp
                else:
                    if len(next_idxs_tmp) > 1:
                        if next_idxs_tmp[1,1] == 0:
                            next_vertex_tmp = subscripts[next_idxs_tmp[1,0],1]
                        else:
                            next_vertex_tmp = subscripts[next_idxs_tmp[1,0],0]
                        previous_vertex = next_vertex
                        next_vertex = next_vertex_tmp
                    else:
                        break
            if next_vertex in junctions:
                edge.append(next_vertex)
                if next_vertex > junctions[ii]:
                    edges_gt.append(edge)
            else:
                edges_gt.append(edge)

    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    graph_gt_img = np.zeros((h,w))
    for ii in range(0,len(edges_gt)):
        mask_start_point = np.ones((h,w))
        mask_end_point = np.ones((h,w))
        edge = edges_gt[ii]
        for jj in range(0,len(edge)-1):
            src = edge[jj]
            src_x = int(np.round(vertices[src,0]))
            src_y = int(np.round(vertices[src,1]))
            target = edge[jj+1]
            target_x = int(np.round(vertices[target,0]))
            target_y = int(np.round(vertices[target,1]))
            line = list(bresenham(src_x, src_y, target_x, target_y))
            for kk in range(0,len(line)):
                graph_gt_img[line[kk][1],line[kk][0]] = 1
            if jj == 0:
                mask_start_point[line[0][1],line[0][0]] = 0
            if jj == len(edge)-2:
                mask_end_point[line[len(line)-1][1],line[len(line)-1][0]] = 0

    return edges_gt, graph_gt_img


visualize_connectivity = True
visualize_errors = False

GT_vessel_segmentation = False
DRIU_vessel_supervision = False
DRIU_point_supervision = False
iterative_graph_creation = True


root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'

if GT_vessel_segmentation:
    results_dir = root_dir + 'test/1st_manual/'
    to_be_cropped = False
    skeletonized = True
elif DRIU_vessel_supervision:
    results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/'
    skeletonized = False
    to_be_cropped = False
elif DRIU_point_supervision:
    results_dir = '/scratch/carlesv/results/DRIVE/Results_DRIU_point_supervision_skeleton/pred_th_130/'
    skeletonized = True
    to_be_cropped = True
else: #iterative
    #results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation/'
    #results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_th_25/'
    #results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset/'
    results_dir = '/scratch/carlesv/results/DRIVE/Results_iterative_graph_creation_no_mask_offset_th_25/'
    to_be_cropped = False
    skeletonized = True

CRR_all = []

for img_idx in range(1,21):

    print(img_idx)

    img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))
    img_array = np.array(img, dtype=np.float32)
    h, w = img_array.shape[:2]

    #Predicted graph with DRIU

    if not GT_vessel_segmentation:

        if DRIU_point_supervision:
            pred = Image.open(results_dir + 'pred_graph_%02d.png' %(img_idx))
        elif iterative_graph_creation:
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_040.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_040_skeleton.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_040_2nd_step.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_100_2nd_step_extended_branches.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_postprocessed_minval_10_maxdepth_20_confidence_th_100_2nd_step_extended_branches_wo_not_connected.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_mask_graph_dilated_skeleton.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_mask_graph_dilated_skeleton_extended_branches_confidence_th_100.png' %(img_idx))
            #pred = Image.open(results_dir + 'pred_graph_%02d_mask_graph_offset_2_dilated_skeleton.png' %(img_idx))
            pred = Image.open(results_dir + 'pred_graph_%02d_mask_graph_offset_2_dilated_skeleton_extended_branches_confidence_th_100.png' %(img_idx))
        else: #DRIU vessel supervision
            pred = Image.open(results_dir + '%02d_test.png' %(img_idx))

        pred = np.array(pred)
        if DRIU_vessel_supervision:
            pred = skeletonize(pred>170)

        G_pred = build_graph(pred)

        print('Predicted graph built')


    #Ground truth graph (Estrada annotations)

    edges_gt, graph_gt_img = extract_edges_from_gt_annotations(img_idx)
    G_gt = build_graph(graph_gt_img)

    print('GT graph built')

    #Ground truth graph from ground truth vessel segmentation

    if GT_vessel_segmentation:

        #results_dir_gt = root_dir + 'test/1st_manual/'
        pred = Image.open(results_dir + '%02d_manual1.gif' %(img_idx))
        pred = np.array(pred)
        pred = skeletonize(pred>200)

        G_pred = build_graph(pred)

        print('GT from vessel segmentation graph built')


    # Find matching between predicted edges and ground truth edges
    CRR = evaluate_connectivity(img_idx, edges_gt, G_gt, pred, G_pred, visualize_connectivity, to_be_cropped)

    print(CRR)

    CRR_all.append(CRR)


print(CRR_all)
print(np.mean(CRR_all))








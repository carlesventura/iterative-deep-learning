__author__ = 'carlesv'
from PIL import Image
import numpy as np
import networkx as nx
import scipy.io as sio

def generate_graph(idx, idx_patch):

    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'
    num_images = 20

    pred = Image.open(results_dir + '%02d_test.png' %(idx))
    pred = np.array(pred)

    f = open('./gt_dbs/DRIVE/vertices_selected.txt','r')
    count = 0

    while count != (idx_patch-1)*num_images + idx-1:
        line = f.readline()
        count += 1

    line = f.readline()
    f.close()

    selected_vertex = int(line.split()[1])

    mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    patch_size = 64
    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(1,pred.shape[0]-1):
        for col_idx in range(1,pred.shape[1]-1):
            node_idx = row_idx*pred.shape[1] + col_idx

            node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx-1,col_idx-1]
            G.add_edge(node_idx,node_topleft_idx,weight=cost)

            node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx-1,col_idx]
            G.add_edge(node_idx,node_top_idx,weight=cost)

            node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx-1,col_idx+1]
            G.add_edge(node_idx,node_topright_idx,weight=cost)

            node_left_idx = row_idx*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx,col_idx-1]
            G.add_edge(node_idx,node_left_idx,weight=cost)

            node_right_idx = row_idx*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx,col_idx+1]
            G.add_edge(node_idx,node_right_idx,weight=cost)

            node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx+1,col_idx-1]
            G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx+1,col_idx]
            G.add_edge(node_idx,node_bottom_idx,weight=cost)

            node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx+1,col_idx+1]
            G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

def generate_graph_center(idx, center):

    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'

    pred = Image.open(results_dir + '%02d_test.png' %(idx))
    pred = np.array(pred)

    patch_size = 64
    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(1,pred.shape[0]-1):
        for col_idx in range(1,pred.shape[1]-1):
            node_idx = row_idx*pred.shape[1] + col_idx

            node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx-1,col_idx-1]
            G.add_edge(node_idx,node_topleft_idx,weight=cost)

            node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx-1,col_idx]
            G.add_edge(node_idx,node_top_idx,weight=cost)

            node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx-1,col_idx+1]
            G.add_edge(node_idx,node_topright_idx,weight=cost)

            node_left_idx = row_idx*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx,col_idx-1]
            G.add_edge(node_idx,node_left_idx,weight=cost)

            node_right_idx = row_idx*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx,col_idx+1]
            G.add_edge(node_idx,node_right_idx,weight=cost)

            node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx+1,col_idx-1]
            G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx+1,col_idx]
            G.add_edge(node_idx,node_bottom_idx,weight=cost)

            node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx+1,col_idx+1]
            G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

def generate_graph_center_connectivity4(idx, center):

    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'

    pred = Image.open(results_dir + '%02d_test.png' %(idx))
    pred = np.array(pred)

    patch_size = 64
    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(1,pred.shape[0]-1):
        for col_idx in range(1,pred.shape[1]-1):
            node_idx = row_idx*pred.shape[1] + col_idx

            node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx-1,col_idx]
            G.add_edge(node_idx,node_top_idx,weight=cost)

            node_left_idx = row_idx*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx,col_idx-1]
            G.add_edge(node_idx,node_left_idx,weight=cost)

            node_right_idx = row_idx*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx,col_idx+1]
            G.add_edge(node_idx,node_right_idx,weight=cost)

            node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx+1,col_idx]
            G.add_edge(node_idx,node_bottom_idx,weight=cost)

    return G

def generate_graph_center_patch_size(idx, center, patch_size):

    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'

    pred = Image.open(results_dir + '%02d_test.png' %(idx))
    pred = np.array(pred)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(1,pred.shape[0]-1):
        for col_idx in range(1,pred.shape[1]-1):
            node_idx = row_idx*pred.shape[1] + col_idx

            node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx-1,col_idx-1]
            G.add_edge(node_idx,node_topleft_idx,weight=cost)

            node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx-1,col_idx]
            G.add_edge(node_idx,node_top_idx,weight=cost)

            node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx-1,col_idx+1]
            G.add_edge(node_idx,node_topright_idx,weight=cost)

            node_left_idx = row_idx*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx,col_idx-1]
            G.add_edge(node_idx,node_left_idx,weight=cost)

            node_right_idx = row_idx*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx,col_idx+1]
            G.add_edge(node_idx,node_right_idx,weight=cost)

            node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
            cost = 255 - pred[row_idx+1,col_idx-1]
            G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
            cost = 255 - pred[row_idx+1,col_idx]
            G.add_edge(node_idx,node_bottom_idx,weight=cost)

            node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
            cost = 255 - pred[row_idx+1,col_idx+1]
            G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

def generate_graph_center_patch_size_min_confidence(idx, center, patch_size, confidence_th):

    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'

    pred = Image.open(results_dir + '%02d_test.png' %(idx))
    pred = np.array(pred)

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    pred = pred[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]

    G=nx.DiGraph()

    for row_idx in range(0,pred.shape[0]):
        for col_idx in range(0,pred.shape[1]):
            node_idx = row_idx*pred.shape[1] + col_idx

            if row_idx > 0 and col_idx > 0:
                node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
                if pred[row_idx-1,col_idx-1] >= confidence_th:
                    cost = 255 - pred[row_idx-1,col_idx-1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_topleft_idx,weight=cost)

            if row_idx > 0:
                node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
                if pred[row_idx-1,col_idx] >= confidence_th:
                    cost = 255 - pred[row_idx-1,col_idx]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_top_idx,weight=cost)

            if row_idx > 0 and col_idx < pred.shape[1]-1:
                node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
                if pred[row_idx-1,col_idx+1] >= confidence_th:
                    cost = 255 - pred[row_idx-1,col_idx+1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_topright_idx,weight=cost)

            if col_idx > 0:
                node_left_idx = row_idx*pred.shape[1] + col_idx-1
                if pred[row_idx,col_idx-1] >= confidence_th:
                    cost = 255 - pred[row_idx,col_idx-1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_left_idx,weight=cost)

            if col_idx < pred.shape[1]-1:
                node_right_idx = row_idx*pred.shape[1] + col_idx+1
                if pred[row_idx,col_idx+1] >= confidence_th:
                    cost = 255 - pred[row_idx,col_idx+1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_right_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx > 0:
                node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
                if pred[row_idx+1,col_idx-1] >= confidence_th:
                    cost = 255 - pred[row_idx+1,col_idx-1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

            if row_idx < pred.shape[0]-1:
                node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
                if pred[row_idx+1,col_idx] >= confidence_th:
                    cost = 255 - pred[row_idx+1,col_idx]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_bottom_idx,weight=cost)

            if row_idx < pred.shape[0]-1 and col_idx < pred.shape[1]-1:
                node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
                if pred[row_idx+1,col_idx+1] >= confidence_th:
                    cost = 255 - pred[row_idx+1,col_idx+1]
                else:
                    cost = 1e8
                G.add_edge(node_idx,node_bottomright_idx,weight=cost)

    return G

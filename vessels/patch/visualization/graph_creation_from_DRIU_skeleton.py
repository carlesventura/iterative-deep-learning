__author__ = 'carlesv'

from PIL import Image
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import operator
from skimage.morphology import skeletonize
import scipy.io as sio
from bresenham import bresenham
from scipy import ndimage


img_idx = 1

root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' % img_idx))
plt.imshow(img)
img_array = np.array(img, dtype=np.float32)
h, w = img_array.shape[:2]

results_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/'

pred = Image.open(results_dir + '%02d_test.png' %(img_idx))
#plt.imshow(pred)
pred = np.array(pred)
pred = skeletonize(pred>100)

G=nx.DiGraph()

for row_idx in range(1,pred.shape[0]-1):
    for col_idx in range(1,pred.shape[1]-1):
        node_idx = row_idx*pred.shape[1] + col_idx

        node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
        if pred[row_idx-1,col_idx-1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_topleft_idx,weight=cost)

        node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
        if pred[row_idx-1,col_idx]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_top_idx,weight=cost)

        node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
        if pred[row_idx-1,col_idx+1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_topright_idx,weight=cost)

        node_left_idx = row_idx*pred.shape[1] + col_idx-1
        if pred[row_idx,col_idx-1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_left_idx,weight=cost)

        node_right_idx = row_idx*pred.shape[1] + col_idx+1
        if pred[row_idx,col_idx+1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_right_idx,weight=cost)

        node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
        if pred[row_idx+1,col_idx-1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

        node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
        if pred[row_idx+1,col_idx]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_bottom_idx,weight=cost)

        node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
        if pred[row_idx+1,col_idx+1]:
            cost = 1
        else:
            cost = 1e10
        G.add_edge(node_idx,node_bottomright_idx,weight=cost)

mask_graph = np.zeros((h,w))
mask_junctions = np.zeros((h,w))

print('Graph built')

#all_paths = nx.all_pairs_dijkstra_path(G)
#print('all_pairs_dijkstra_path computed')

start_row = 251
start_col = 83
source_idx = start_row*pred.shape[1] + start_col

#length, paths = nx.single_source_dijkstra(G, source_idx, cutoff=500, weight='weight')
length, paths = nx.single_source_dijkstra(G, source_idx, cutoff=1e8, weight='weight')

print('single_source_dijkstra computed')

# pos_y_vector = []
# pos_x_vector = []
# for node_idx in paths.keys():
#     pos_y = paths[node_idx][-1] / pred.shape[1]
#     pos_x = paths[node_idx][-1] % pred.shape[1]
#     pos_y_vector.append(pos_y)
#     pos_x_vector.append(pos_x)
#
# plt.scatter(pos_x_vector,pos_y_vector,color='blue',marker='+')
#
# plt.show()

avg_length = {}
for node_idx in paths.keys():
    #avg_length[node_idx] = float(length[node_idx]+1)/(len(paths[node_idx])*len(paths[node_idx]))
    avg_length[node_idx] = float(length[node_idx]+1)/np.exp(len(paths[node_idx]))

print('average cost computed')

sorted_keys = sorted(avg_length.items(), key=operator.itemgetter(1))

print('paths sorted by averaged length')

colors = ['red', 'blue', 'cyan', 'green', 'purple']
count = 0
offset = 0

segments_x = []
segments_y = []

min_length = 5

for ii in range(0,len(sorted_keys)):
    key = sorted_keys[ii][0]
    pos_y_vector = []
    pos_x_vector = []
    for jj in range(0,len(paths[key])):
        idx = len(paths[key])-1-jj
        pos_y = paths[key][idx] / pred.shape[1]
        pos_x = paths[key][idx] % pred.shape[1]
        if mask_graph[pos_y,pos_x] == 0:
            pos_y_vector.append(pos_y)
            pos_x_vector.append(pos_x)
            #mask_graph[pos_y,pos_x] = 1
        else:
            pos_y_vector.append(pos_y)
            pos_x_vector.append(pos_x)
            if len(pos_y_vector) > min_length:
                mask_junctions[pos_y,pos_x] = 1
            break

    if len(pos_y_vector) > min_length:
        segments_x.append(pos_x_vector)
        segments_y.append(pos_y_vector)
        for kk in range(0,len(pos_y_vector)):
            #mask_graph[pos_y_vector[kk],pos_x_vector[kk]] = 1
            mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
        plt.scatter(pos_x_vector,pos_y_vector,color=colors[count%5],marker='.')
        plt.scatter(pos_x_vector[0],pos_y_vector[0],color='black',marker='o')
        plt.scatter(pos_x_vector[-1],pos_y_vector[-1],color='black',marker='+')
        count += 1

plt.show()

#split segments into graph edges

edges_x = []
edges_y = []
dict_junctions = {}
dict_edges = {}
edge_id = 0
edge_ids_matrix = -1*np.ones((h,w))
for ii in range(0,len(segments_x)):
    pos_y_vector = []
    pos_x_vector = []
    junction_pos = segments_y[ii][0]*w + segments_x[ii][0]
    dict_junctions[junction_pos] = [edge_id]
    dict_edges[edge_id] = [junction_pos]
    for jj in range(1,len(segments_x[ii])):
        if not mask_junctions[segments_y[ii][jj],segments_x[ii][jj]]:
            pos_y_vector.append(segments_y[ii][jj])
            pos_x_vector.append(segments_x[ii][jj])
            edge_ids_matrix[segments_y[ii][jj],segments_x[ii][jj]] = edge_id
        else:
            edges_y.append(pos_y_vector)
            edges_x.append(pos_x_vector)
            junction_pos = segments_y[ii][jj]*w + segments_x[ii][jj]
            if junction_pos in dict_junctions.keys():
                dict_junctions[junction_pos].append(edge_id)
            else:
                dict_junctions[junction_pos] = [edge_id]
            dict_edges[edge_id].append(junction_pos)
            if jj < len(segments_x[ii])-1:
                dict_junctions[junction_pos].append(edge_id+1)
                dict_edges[edge_id+1] = [junction_pos]
            pos_y_vector = []
            pos_x_vector = []
            edge_id += 1

num_edges = len(edges_x)
print('segments split into edges')

junction_map_pos = {}
junction_map_id = {}
count_junctions = 0
for junction_pos in dict_junctions.keys():
    junction_map_pos[junction_pos] = count_junctions
    junction_map_id[count_junctions] = junction_pos
    count_junctions += 1

adjacency_vertices_matrix = np.zeros((count_junctions,count_junctions))

for edge_id in dict_edges.keys():
    source = junction_map_pos[dict_edges[edge_id][0]]
    target = junction_map_pos[dict_edges[edge_id][1]]
    adjacency_vertices_matrix[source,target] = edge_id+1
    adjacency_vertices_matrix[target,source] = edge_id+1

print('adjacency_vertices_matrix computed')

adjacency_edge_matrix = np.zeros((num_edges,num_edges))

for junction_id in dict_junctions.keys():
    for ii in range(0,len(dict_junctions[junction_id])-1):
        for jj in range(ii+1,len(dict_junctions[junction_id])):
            adjacency_edge_matrix[dict_junctions[junction_id][ii],dict_junctions[junction_id][jj]] = junction_id+1
            adjacency_edge_matrix[dict_junctions[junction_id][jj],dict_junctions[junction_id][ii]] = junction_id+1

print('adjacency_edge_matrix computed')


#Ground truth graph

mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %img_idx)
vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
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


# Find matching between predicted edges and ground truth edges

graph_gt_img = np.zeros((h,w))
for ii in range(0,len(edges_gt)):
    mask = np.ones((h,w))
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
            mask[line[kk][1],line[kk][0]] = 0
            graph_gt_img[line[kk][1],line[kk][0]] = 1
    dist = ndimage.distance_transform_edt(mask)
    indxs = np.argwhere(dist<5)
    exploring_mask = np.zeros((h,w))
    exploring_mask[indxs[:,0],indxs[:,1]]=1
    edges_in_mask = np.unique(edge_ids_matrix[indxs[:,0],indxs[:,1]])
    if ii%20 == 0:
        print(edges_in_mask)
        plt.imshow(mask)
        for jj in range(1,len(edges_in_mask)):
            edge_x = edges_x[int(edges_in_mask[jj])]
            edge_y = edges_y[int(edges_in_mask[jj])]
            plt.scatter(edge_x,edge_y)
        plt.show()
#scipy.misc.imsave('/scratch/carlesv/gt_dbs/DRIVE/test/graph_annotations/gt_graph_%02d.png' % img_idx, graph_gt_img)





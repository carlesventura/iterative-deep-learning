__author__ = 'carlesv'

from PIL import Image
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import operator

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

G=nx.DiGraph()

for row_idx in range(1,pred.shape[0]-1):
    for col_idx in range(1,pred.shape[1]-1):
        node_idx = row_idx*pred.shape[1] + col_idx

        node_topleft_idx = (row_idx-1)*pred.shape[1] + col_idx-1
        cost = 255 - pred[row_idx-1,col_idx-1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_topleft_idx,weight=cost)

        node_top_idx = (row_idx-1)*pred.shape[1] + col_idx
        cost = 255 - pred[row_idx-1,col_idx]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_top_idx,weight=cost)

        node_topright_idx = (row_idx-1)*pred.shape[1] + col_idx+1
        cost = 255 - pred[row_idx-1,col_idx+1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_topright_idx,weight=cost)

        node_left_idx = row_idx*pred.shape[1] + col_idx-1
        cost = 255 - pred[row_idx,col_idx-1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_left_idx,weight=cost)

        node_right_idx = row_idx*pred.shape[1] + col_idx+1
        cost = 255 - pred[row_idx,col_idx+1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_right_idx,weight=cost)

        node_bottomleft_idx = (row_idx+1)*pred.shape[1] + col_idx-1
        cost = 255 - pred[row_idx+1,col_idx-1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

        node_bottom_idx = (row_idx+1)*pred.shape[1] + col_idx
        cost = 255 - pred[row_idx+1,col_idx]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_bottom_idx,weight=cost)

        node_bottomright_idx = (row_idx+1)*pred.shape[1] + col_idx+1
        cost = 255 - pred[row_idx+1,col_idx+1]
        if cost > 200:
            cost = 1e10
        G.add_edge(node_idx,node_bottomright_idx,weight=cost)

mask_graph = np.zeros((h,w))

print('Graph built')

#all_paths = nx.all_pairs_dijkstra_path(G)
#print('all_pairs_dijkstra_path computed')

start_row = 253
start_col = 81
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
offset = 4
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
            break

    if len(pos_y_vector) > 5:
        for kk in range(0,len(pos_y_vector)):
            #mask_graph[pos_y_vector[kk],pos_x_vector[kk]] = 1
            mask_graph[pos_y_vector[kk]-offset:pos_y_vector[kk]+offset+1,pos_x_vector[kk]-offset:pos_x_vector[kk]+offset+1] = 1
        plt.scatter(pos_x_vector,pos_y_vector,color=colors[count%5],marker='.')
        plt.scatter(pos_x_vector[0],pos_y_vector[0],color='black',marker='o')
        plt.scatter(pos_x_vector[-1],pos_y_vector[-1],color='black',marker='+')
        count += 1

plt.show()

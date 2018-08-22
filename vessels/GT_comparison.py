__author__ = 'carlesv'
import scipy.io as sio
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from PIL import Image


def read_gt_points_Estrada(idx):

    if idx > 20:
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' % (idx))
    else:
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' % (idx))
    junctions_ids = np.squeeze(mat_contents['G']['junctions'][0,0])
    all_vertices = mat_contents['G']['V'][0,0]-1

    gt_points_Estrada = all_vertices[junctions_ids-1,:]

    return gt_points_Estrada

def read_gt_points_Azzopardi(idx):
    annotation_file = '/scratch_net/boxy/carlesv/RetinalFeatures/%02d_manual1_gt.txt' % (idx)
    file = open(annotation_file, 'r')
    lines = file.readlines()
    file.close()
    x_fullres = np.nan * np.zeros((len(lines), 2))
    for i in range(0,len(lines)):
        x_fullres[i, 0] = int(lines[i].split(',')[1])
        x_fullres[i, 1] = int(lines[i].split(',')[0])
    return x_fullres


dist_th = 64
num_images = 40

precision_all = []
recall_all = []

for idx in range(1,num_images+1):

    gt_points_Azzopardi = read_gt_points_Azzopardi(idx)
    gt_points_Estrada = read_gt_points_Estrada(idx)

    if idx >20:
        gt_vessels = Image.open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/training/1st_manual/%02d_manual1.gif' % idx)
    else:
        gt_vessels = Image.open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/test/1st_manual/%02d_manual1.gif' % idx)

    cost = np.zeros((gt_points_Azzopardi.shape[0],gt_points_Estrada.shape[0]),np.float32)

    for i in range(0,gt_points_Azzopardi.shape[0]):
        for j in range(0,gt_points_Estrada.shape[0]):
            dist = (gt_points_Azzopardi[i,0]-gt_points_Estrada[j,0])*(gt_points_Azzopardi[i,0]-gt_points_Estrada[j,0])+(gt_points_Azzopardi[i,1]-gt_points_Estrada[j,1])*(gt_points_Azzopardi[i,1]-gt_points_Estrada[j,1])
            if dist > dist_th:
                dist = 1000
            cost[i,j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    plt.imshow(gt_vessels)
    plt.plot(gt_points_Estrada[:,0], gt_points_Estrada[:,1], ls='none', color='green',marker='+', ms=10, lw=1.5)
    plt.plot(gt_points_Azzopardi[:,0], gt_points_Azzopardi[:,1], ls='none', color='red',marker='o', ms=10, lw=1.5, mfc='none')
    for i in range(0,len(row_ind)):
        if cost[row_ind[i],col_ind[i]] < 1000:
            plt.plot([gt_points_Azzopardi[row_ind[i],0], gt_points_Estrada[col_ind[i],0]], [gt_points_Azzopardi[row_ind[i],1], gt_points_Estrada[col_ind[i],1]],color='blue')
    plt.show()

    total_detections = gt_points_Azzopardi.shape[0]
    true_positives = 0

    for i in range(0,len(row_ind)):
            if cost[row_ind[i],col_ind[i]] < 1000:
                true_positives += 1

    false_positives = total_detections - true_positives

    if total_detections > 0:
        precision = float(true_positives) / total_detections
    else:
        precision = 1

    recall = float(true_positives) / gt_points_Estrada.shape[0]

    precision_all.append(precision)
    recall_all.append(recall)

recall_overall = np.mean(recall_all,axis=0)
precision_overall = np.mean(precision_all,axis=0)

F_overall = np.divide(2*np.multiply(recall_overall,precision_overall),np.add(recall_overall,precision_overall))
F_max = np.max(F_overall)

print(precision_all)
print(recall_all)
print(precision_overall)
print(recall_overall)
print(F_max)






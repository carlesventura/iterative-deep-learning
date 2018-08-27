__author__ = 'carlesv'
import matplotlib.pyplot as plt
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString
import scipy.io as sio

dist_th = 4
num_images = 20
num_patches = 50
start_img = 1
patch_size = 64

gt_base_dir = './results_dir_vessels/gt_test'
results_base_dir = './results_dir_vessels/results'

configs = ['_not_connected', '_connected', '_connected_same_vessel', '_connected_same_vessel_wo_bifurcations']

idx_patch = 1
num_patches = 50

f = open('./gt_dbs/DRIVE/vertices_selected.txt','r')
count = 0

while count != (idx_patch-1)*num_images + start_img-1:
    line = f.readline()
    count += 1

for idx in range(start_img,start_img+num_images):

    fig, axes = plt.subplots(5, 4)

    mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
    art = np.squeeze(mat_contents['G']['art'][0,0])
    ven = np.squeeze(mat_contents['G']['ven'][0,0])

    line = f.readline()
    selected_vertex = int(line.split()[1])
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)

    for ii in range(0,len(subscripts)):
        segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
        xcoords, ycoords = segment.xy
        if art[subscripts[ii,0]-1] and art[subscripts[ii,1]-1]:
            axes[1,0].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='red', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,1].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='red', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,2].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='red', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,3].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='red', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
        else:
            axes[1,0].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='blue', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,1].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='blue', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,2].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='blue', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
            axes[1,3].plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='blue', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)

    axes[1,0].set_xlim([0, patch_size-1])
    axes[1,0].set_ylim([patch_size-1,0])
    axes[1,0].set_aspect(1)
    axes[1,1].set_xlim([0, patch_size-1])
    axes[1,1].set_ylim([patch_size-1,0])
    axes[1,1].set_aspect(1)
    axes[1,2].set_xlim([0, patch_size-1])
    axes[1,2].set_ylim([patch_size-1,0])
    axes[1,2].set_aspect(1)
    axes[1,3].set_xlim([0, patch_size-1])
    axes[1,3].set_ylim([patch_size-1,0])
    axes[1,3].set_aspect(1)

    for config_id in range(0,len(configs)):

        config_type = configs[config_id]

        precision_patch = np.zeros(254,np.float32)
        recall_patch = np.zeros(254,np.float32)

        retina_img = Image.open(gt_base_dir + config_type + '/img_%02d_patch_%02d_img.png' %(idx, idx_patch))
        pred = np.load(results_base_dir + config_type + '/epoch_1800/img_%02d_patch_%02d.npy' %(idx, idx_patch))

        axes[0,config_id].imshow(retina_img)

        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)
        positions = (sources['x_peak'], sources['y_peak'])

        axes[2,config_id].imshow(pred, interpolation='nearest')
        axes[2,config_id].plot(sources['x_peak'], sources['y_peak'], ls='none', color='red',marker='+', ms=10, lw=1.5)

        gt_img = Image.open(gt_base_dir + config_type + '/img_%02d_patch_%02d_gt.png' %(idx, idx_patch))

        mean_gt, median_gt, std_gt = sigma_clipped_stats(gt_img, sigma=3.0)
        threshold_gt = median_gt + (10.0 * std_gt)
        sources_gt = find_peaks(np.array(gt_img), threshold_gt, box_size=3)
        if len(sources_gt) == 0:
            gt_points = []
        else:
            gt_points = (sources_gt['x_peak'], sources_gt['y_peak'])

        for peak_th in range(1,255):

            valid_peaks = sources[sources['peak_value'] > peak_th]
            positions = (valid_peaks['x_peak'], valid_peaks['y_peak'])

            if len(sources_gt) > 0:

                cost = np.zeros((len(positions[0]),len(gt_points[0])),np.float32)

                for i in range(0,len(positions[0])):
                    for j in range(0,len(gt_points[0])):
                        dist = (positions[0][i]-gt_points[0][j])*(positions[0][i]-gt_points[0][j])+(positions[1][i]-gt_points[1][j])*(positions[1][i]-gt_points[1][j])
                        if dist > dist_th:
                            dist = 1000
                        cost[i,j] = dist

                row_ind, col_ind = linear_sum_assignment(cost)

                total_detections = len(positions[0])
                true_positives = 0

                for i in range(0,len(row_ind)):
                        if cost[row_ind[i],col_ind[i]] < 1000:
                            true_positives += 1

                if peak_th == 1:
                    axes[3,config_id].imshow(gt_img)
                    axes[3,config_id].plot(gt_points[0], gt_points[1], ls='none', color='green',marker='o', ms=10, lw=1.5, mfc='none')
                    axes[3,config_id].plot(positions[0], positions[1], ls='none', color='red',marker='+', ms=10, lw=1.5)
                    for i in range(0,len(row_ind)):
                        if cost[row_ind[i],col_ind[i]] < 1000:
                            axes[3,config_id].plot([positions[0][row_ind[i]], gt_points[0][col_ind[i]]], [positions[1][row_ind[i]], gt_points[1][col_ind[i]]],color='blue')

                false_positives = total_detections - true_positives

                if total_detections > 0:
                    precision = float(true_positives) / total_detections
                else:
                    precision = 1

                recall = float(true_positives) / len(gt_points[0])

                precision_patch[peak_th-1] = precision
                recall_patch[peak_th-1] = recall


                if peak_th == 254:
                    axes[4,config_id].plot(recall_patch,precision_patch)
                    axes[4,config_id].set_xlim([0,1])
                    axes[4,config_id].set_ylim([0,1])
                    axes[4,config_id].set_aspect(1)
                    if config_id == len(configs)-1:
                        plt.show()

            else: # len(sources_gt) = 0
                total_detections = len(positions[0])
                true_positives = 0
                false_positives = total_detections - true_positives

                if peak_th == 1:
                    axes[3,config_id].imshow(gt_img)
                    axes[3,config_id].plot(positions[0], positions[1], ls='none', color='red',marker='+', ms=10, lw=1.5)

                if total_detections > 0:
                    precision = 0
                else:
                    precision = 1

                recall = 1

                precision_patch[peak_th-1] = precision
                recall_patch[peak_th-1] = recall

                if peak_th == 254:
                    axes[4,config_id].plot(recall_patch,precision_patch)
                    axes[4,config_id].set_xlim([0,1])
                    axes[4,config_id].set_ylim([0,1])
                    axes[4,config_id].set_aspect(1)
                    if config_id == len(configs)-1:
                        plt.show()



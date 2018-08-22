__author__ = 'carlesv'
import matplotlib.pyplot as plt
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.io as sio
from scipy.cluster.hierarchy import fcluster, linkage
from astropy.table import Table

dist_th = 5
num_images = 20
num_patches = 50
start_img = 1

connected = True
from_same_vessel = True
bifurcations_allowed = False

def valid_sources(sources):

    samples = np.zeros((len(sources),2),int)
    for ii in range(0,len(sources)):
        samples[ii,0] = sources['x_peak'][ii]
        samples[ii,1] = sources['y_peak'][ii]

    Z = linkage(samples, 'single', 'cityblock')
    max_d = 1
    clusters = fcluster(Z, max_d, criterion='distance')
    clusters_visited = []
    clustered_sources = {}
    clustered_sources['x_peak'] = []
    clustered_sources['y_peak'] = []
    clustered_sources['peak_value'] = []

    for ii in range(0, len(clusters)):
        cluster_idx = clusters[ii]
        if cluster_idx not in clusters_visited:
            clusters_visited.append(cluster_idx)
            sample_idxs = np.argwhere(clusters == cluster_idx)
            mean_x = np.mean(sources['x_peak'][sample_idxs])
            mean_y = np.mean(sources['y_peak'][sample_idxs])
            mean_peak = np.mean(sources['peak_value'][sample_idxs])
            clustered_sources['x_peak'].append(mean_x)
            clustered_sources['y_peak'].append(mean_y)
            clustered_sources['peak_value'].append(mean_peak)

    clustered_sources = Table([clustered_sources['x_peak'], clustered_sources['y_peak'], clustered_sources['peak_value']], names=('x_peak', 'y_peak', 'peak_value'))

    return clustered_sources


results_dir_DRIU = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/'

if not connected:
    gt_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_not_connected/'
    results_dir_SHG = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_not_connected/'
else:
    if from_same_vessel:
        if bifurcations_allowed:
            gt_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected_same_vessel/'
            results_dir_SHG = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected_same_vessel/'
        else:
            gt_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected_same_vessel_wo_bifurcations/'
            results_dir_SHG = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected_same_vessel_wo_bifurcations/'
    else:
        gt_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected/'
        results_dir_SHG = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_connected/'


precision_all = np.zeros((num_images*num_patches,254),np.float32)
recall_all = np.zeros((num_images*num_patches,254),np.float32)

for idx in range(start_img,start_img+num_images):
    for idx_patch in range(1,num_patches+1):

        pred_DRIU = Image.open(results_dir_DRIU + '%02d_test.png' %(idx))
        pred_DRIU = np.array(pred_DRIU)

        f = open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/vertices_selected_margin_6.txt','r')
        count = 0

        while count != (idx_patch-1)*num_images + idx-1:
            line = f.readline()
            count += 1

        line = f.readline()
        f.close()

        selected_vertex = int(line.split()[1])

        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
        vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
        center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

        patch_size = 64
        x_tmp = int(center[0]-patch_size/2)
        y_tmp = int(center[1]-patch_size/2)

        pred_DRIU = pred_DRIU[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size]
        margin = int(np.round(patch_size/10.0))
        pred_DRIU[0:margin,:] = 0
        pred_DRIU[margin:patch_size-margin,0:margin] = 0
        pred_DRIU[margin:patch_size-margin,patch_size-margin:patch_size] = 0
        pred_DRIU[patch_size-margin:patch_size,:] = 0
        pred_DRIU[margin+1:patch_size-margin-1,margin+1:patch_size-margin-1] = 0


        #mean, median, std = sigma_clipped_stats(img, sigma=3.0)
        mean, median, std = sigma_clipped_stats(pred_DRIU, sigma=3.0)
        threshold = median + (10.0 * std)
        #sources = find_peaks(np.array(img), threshold, box_size=3)
        sources = find_peaks(pred_DRIU, threshold, box_size=3)
        sources = valid_sources(sources)
        positions = (sources['x_peak'], sources['y_peak'])

        pred_SHG = np.load(results_dir_SHG + 'epoch_1800/img_%02d_patch_%02d.npy' %(idx, idx_patch))

        for ii in range(0,len(sources)):
            tmp_x = sources['x_peak'][ii]
            tmp_y = sources['y_peak'][ii]
            max_SGR_val = np.max(pred_SHG[int(tmp_y)-1:int(tmp_y)+2, int(tmp_x)-1:int(tmp_x)+2])
            #if sources['peak_value'][ii] > 1:
            #    sources['peak_value'][ii] = np.max([max_SGR_val, 2])
            #sources['peak_value'][ii] = np.max([max_SGR_val, sources['peak_value'][ii]])
            #sources['peak_value'][ii] = np.average([max_SGR_val, sources['peak_value'][ii]])
            sources['peak_value'][ii] = 0.8*max_SGR_val + 0.2*sources['peak_value'][ii]


        gt_img = Image.open(gt_dir + 'img_%02d_patch_%02d_gt.png' %(idx, idx_patch))
        mean_gt, median_gt, std_gt = sigma_clipped_stats(gt_img, sigma=3.0)
        threshold_gt = median_gt + (10.0 * std_gt)
        sources_gt = find_peaks(np.array(gt_img), threshold_gt, box_size=3)

        if len(sources_gt) == 0:
            gt_points = []
        else:
            if len(sources_gt) > 1:
                sources_gt = valid_sources(sources_gt)
            gt_points = (sources_gt['x_peak'], sources_gt['y_peak'])


        #peak_th = 10
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



                false_positives = total_detections - true_positives

                if total_detections > 0:
                    precision = float(true_positives) / total_detections
                else:
                    precision = 1

                recall = float(true_positives) / len(gt_points[0])

                #precision_all[idx-1,peak_th-10] = precision
                #recall_all[idx-1,peak_th-10] = recall

                precision_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-1] = precision
                recall_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-1] = recall


            else: # len(sources_gt) = 0
                total_detections = len(positions[0])
                true_positives = 0
                false_positives = total_detections - true_positives

                if total_detections > 0:
                    precision = 0
                else:
                    precision = 1

                recall = 1

                precision_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-1] = precision
                recall_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-1] = recall


recall_overall = np.mean(recall_all,axis=0)
precision_overall = np.mean(precision_all,axis=0)

F_overall = np.divide(2*np.multiply(recall_overall,precision_overall),np.add(recall_overall,precision_overall))
F_max = np.max(F_overall)
F_max_idx = np.argmax(F_overall)
recall_F_max = recall_overall[F_max_idx]
precision_F_max = precision_overall[F_max_idx]

if not connected:
    output_file = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/PR_SGH_from_DRIU_not_connected.npz'
else:
    if from_same_vessel:
        if bifurcations_allowed:
            output_file = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/PR_SGH_from_DRIU_connected_same_vessel.npz'
        else:
            output_file = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/PR_SGH_from_DRIU_connected_same_vessel_wo_bifurcations.npz'
    else:
        output_file = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/PR_SGH_from_DRIU_results_connected.npz'

np.savez(output_file, recall_overall=recall_overall, precision_overall=precision_overall, recall_F_max=recall_F_max, precision_F_max=precision_F_max )

print(F_max)
print(precision_F_max)
print(recall_F_max)

plt.figure()
plt.plot(recall_overall,precision_overall)
plt.plot(recall_F_max,precision_F_max,color='red',marker='+', ms=10)
plt.ylim([0,1])
plt.xlim([0,1])
ax = plt.gca()
ax.set_aspect(1)
plt.show()


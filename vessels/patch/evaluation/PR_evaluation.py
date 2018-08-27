__author__ = 'carlesv'
import matplotlib.pyplot as plt
from photutils import CircularAperture
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from photutils import find_peaks
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.io as sio
from scipy.cluster.hierarchy import fcluster, linkage
from astropy.table import Table

star_finder = False
see_plots = False
dist_th = 5
num_images = 20
num_patches = 50
start_img = 1
epoch = 1800

DRIU_baseline = False
junctions = False
connected = True
from_same_vessel = False
bifurcations_allowed = True
save_results = False

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

if DRIU_baseline:
    results_dir = './results_dir_vessels/results_DRIU_vessel_segmentation/'
    if not connected:
        gt_dir = './results_dir_vessels/gt_test_not_connected/'
    else:
        if from_same_vessel:
            if bifurcations_allowed:
                gt_dir = './results_dir_vessels/gt_test_connected_same_vessel/'
            else:
                gt_dir = './results_dir_vessels/gt_test_connected_same_vessel_wo_bifurcations/'
        else:
            gt_dir = './results_dir_vessels/gt_test_connected/'
else:
    if junctions:
        gt_dir = './results_dir_vessels/gt_test_junctions/'
        results_dir = './results_dir_vessels/results_junctions/'
    else:
        if not connected:
            gt_dir = './results_dir_vessels/gt_test_not_connected/'
            results_dir = './results_dir_vessels/results_not_connected/'
        else:
            if from_same_vessel:
                if bifurcations_allowed:
                    gt_dir = './results_dir_vessels/gt_test_connected_same_vessel/'
                    results_dir = './results_dir_vessels/results_connected_same_vessel/'
                else:
                    gt_dir = './results_dir_vessels/gt_test_connected_same_vessel_wo_bifurcations/'
                    results_dir = './results_dir_vessels/results_connected_same_vessel_wo_bifurcations/'
            else:
                gt_dir = './results_dir_vessels/gt_test_connected/'
                results_dir = './results_dir_vessels/results_connected/'

low_peak_th = 1
high_peak_th = 255

precision_all = np.zeros((num_images*num_patches,high_peak_th-low_peak_th),np.float32)
recall_all = np.zeros((num_images*num_patches,high_peak_th-low_peak_th),np.float32)

count_no_points_gt = 0

for idx in range(start_img,start_img+num_images):
    for idx_patch in range(1,num_patches+1):


        retina_img = Image.open(gt_dir + 'img_%02d_patch_%02d_img.png' %(idx, idx_patch))

        if DRIU_baseline:
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
            margin = int(np.round(patch_size/10.0))
            pred[0:margin,:] = 0
            pred[margin:patch_size-margin,0:margin] = 0
            pred[margin:patch_size-margin,patch_size-margin:patch_size] = 0
            pred[patch_size-margin:patch_size,:] = 0
            pred[margin+1:patch_size-margin-1,margin+1:patch_size-margin-1] = 0

        else:
            pred = np.load(results_dir + 'epoch_' + str(epoch) + '/img_%02d_patch_%02d.npy' %(idx, idx_patch))

        if see_plots and idx_patch==1:
            fig, axes = plt.subplots(2, 2)
            axes[0,0].imshow(retina_img)

        if star_finder:
            mean, median, std = sigma_clipped_stats(pred, sigma=3.0, iters=5)
            threshold = median + (10.0 * std)
            daofind = DAOStarFinder(fwhm=2, threshold=threshold)
            sources = daofind(pred - median)
            positions = (sources['xcentroid'], sources['ycentroid'])
            apertures = CircularAperture(positions, r=4.)
            if see_plots and idx_patch==1:
                axes[0,1].imshow(pred, interpolation='nearest')
                axes[0,1].plot(sources['xcentroid'], sources['ycentroid'], ls='none', color='red',marker='+', ms=10, lw=1.5)
        else:
            mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
            threshold = median + (10.0 * std)
            sources = find_peaks(pred, threshold, box_size=3)

            if DRIU_baseline:
                sources = valid_sources(sources)

            positions = (sources['x_peak'], sources['y_peak'])
            if see_plots and idx_patch==1:
                axes[0,1].imshow(pred, interpolation='nearest')
                axes[0,1].plot(sources['x_peak'], sources['y_peak'], ls='none', color='red',marker='+', ms=10, lw=1.5)

        gt_img = Image.open(gt_dir + 'img_%02d_patch_%02d_gt.png' %(idx, idx_patch))
        mean_gt, median_gt, std_gt = sigma_clipped_stats(gt_img, sigma=3.0)
        threshold_gt = median_gt + (10.0 * std_gt)
        sources_gt = find_peaks(np.array(gt_img), threshold_gt, box_size=3)

        if len(sources_gt) == 0:
            gt_points = []
            count_no_points_gt = count_no_points_gt + 1
        else:
            if len(sources_gt) > 1:
                sources_gt = valid_sources(sources_gt)
            gt_points = (sources_gt['x_peak'], sources_gt['y_peak'])

        for peak_th in range(low_peak_th,high_peak_th):

            if star_finder:
                valid_peaks = sources[sources['peak'] > peak_th]
                positions = (valid_peaks['xcentroid'], valid_peaks['ycentroid'])
            else:
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

                if see_plots and peak_th == low_peak_th and idx_patch==1:

                    axes[1,0].imshow(gt_img)
                    axes[1,0].plot(gt_points[0], gt_points[1], ls='none', color='green',marker='o', ms=10, lw=1.5, mfc='none')
                    axes[1,0].plot(positions[0], positions[1], ls='none', color='red',marker='+', ms=10, lw=1.5)
                    for i in range(0,len(row_ind)):
                        if cost[row_ind[i],col_ind[i]] < 1000:
                            axes[1,0].plot([positions[0][row_ind[i]], gt_points[0][col_ind[i]]], [positions[1][row_ind[i]], gt_points[1][col_ind[i]]],color='blue')


                false_positives = total_detections - true_positives

                if total_detections > 0:
                    precision = float(true_positives) / total_detections
                else:
                    precision = 1

                recall = float(true_positives) / len(gt_points[0])

                precision_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-low_peak_th] = precision
                recall_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-low_peak_th] = recall

                if see_plots and peak_th == (high_peak_th-1) and idx_patch==1:
                    axes[1,1].plot(recall_all[(idx-start_img)*num_patches+idx_patch-1,:],precision_all[(idx-start_img)*num_patches+idx_patch-1,:])
                    axes[1,1].set_xlim([0,1])
                    axes[1,1].set_ylim([0,1])
                    axes[1,1].set_aspect(1)
                    plt.show(block=False)

            else: # len(sources_gt) = 0
                total_detections = len(positions[0])
                true_positives = 0
                false_positives = total_detections - true_positives

                if see_plots and peak_th == low_peak_th and idx_patch==1:

                    axes[1,0].imshow(gt_img)
                    axes[1,0].plot(positions[0], positions[1], ls='none', color='red',marker='+', ms=10, lw=1.5)

                if total_detections > 0:
                    precision = 0
                else:
                    precision = 1

                recall = 1

                precision_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-low_peak_th] = precision
                recall_all[(idx-start_img)*num_patches+idx_patch-1,peak_th-low_peak_th] = recall

                if see_plots and peak_th == (high_peak_th-1) and idx_patch==1:
                    axes[1,1].plot(recall_all[(idx-start_img)*num_patches+idx_patch-1,:],precision_all[(idx-start_img)*num_patches+idx_patch-1,:])
                    axes[1,1].set_xlim([0,1])
                    axes[1,1].set_ylim([0,1])
                    axes[1,1].set_aspect(1)
                    plt.show(block=False)


recall_overall = np.mean(recall_all,axis=0)
precision_overall = np.mean(precision_all,axis=0)

F_overall = np.divide(2*np.multiply(recall_overall,precision_overall),np.add(recall_overall,precision_overall))

print(recall_overall)
print(precision_overall)
print(F_overall)

F_max = np.max(F_overall)
F_max_idx = np.argmax(F_overall)
recall_F_max = recall_overall[F_max_idx]
precision_F_max = precision_overall[F_max_idx]

print(count_no_points_gt)
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

if save_results:
    if DRIU_baseline:
        if not connected:
            output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_not_connected.npz'
        else:
            if from_same_vessel:
                if bifurcations_allowed:
                    output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_connected_same_vessel.npz'
                else:
                    output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_connected_same_vessel_wo_bifurcations.npz'
            else:
                output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_results_connected.npz'
    else:
        if junctions:
            output_file = './results_dir_vessels/PR_junctions.npz'
        else:
            if not connected:
                output_file = './results_dir_vessels/PR_not_connected.npz'
            else:
                if from_same_vessel:
                    if bifurcations_allowed:
                        output_file = './results_dir_vessels/PR_connected_same_vessel.npz'
                    else:
                        output_file = './results_dir_vessels/PR_connected_same_vessel_wo_bifurcations.npz'
                else:
                    output_file = './results_dir_vessels/PR_results_connected.npz'


    np.savez(output_file, recall_overall=recall_overall, precision_overall=precision_overall, recall_F_max=recall_F_max, precision_F_max=precision_F_max )


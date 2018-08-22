__author__ = 'carlesv'
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


iterative = True
ground_truth = False

root_dir='/scratch_net/boxy/carlesv/gt_dbs/MassachusettsRoads/test/images/'
test_img_filenames = os.listdir(root_dir)

if ground_truth:
    output_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6_200_epoches/results_gt/'
    pred_results = '/scratch_net/boxy/carlesv/gt_dbs/MassachusettsRoads/test/1st_manual_skeletons/'
else:
    if iterative:
        output_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6_200_epoches/results_final_th_20/'
        pred_results = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/iterative_results_prediction_130_vgg_th_20_local_mask_skeleton/'
    else:
        output_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6_200_epoches/results_vgg_th_150/'
        pred_results = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/test_results_vgg_skeletons/th_150/'

for img_idx in range(0,len(test_img_filenames)):

    img_filename = test_img_filenames[img_idx]
    img = Image.open(os.path.join(root_dir, img_filename))
    img = np.array(img, dtype=np.uint8)

    #plt.figure(figsize=(12,12), dpi=60)
    plt.figure(figsize=(6,6), dpi=120)
    plt.imshow(img)

    mask_graph_skeleton = Image.open(os.path.join(pred_results, img_filename))
    mask_graph_skeleton = np.array(mask_graph_skeleton, dtype=np.uint8)

    indxs_skel = np.argwhere(mask_graph_skeleton==255)
    plt.scatter(indxs_skel[:,1],indxs_skel[:,0],color='red',marker='+')
    plt.axis('off')
    #plt.savefig(output_dir + img_filename[:-4] + 'png', bbox_inches='tight')
    #plt.close()
    plt.show()














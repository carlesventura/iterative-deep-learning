__author__ = 'carlesv'
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


root_dir='./gt_dbs/MassachusettsRoads/test/images/'
test_img_filenames = os.listdir(root_dir)

pred_results = './results_dir/iterative_results_prediction_vgg/'

for img_idx in range(0,len(test_img_filenames)):

    img_filename = test_img_filenames[img_idx]
    img = Image.open(os.path.join(root_dir, img_filename))
    img = np.array(img, dtype=np.uint8)

    plt.figure(figsize=(6,6), dpi=120)
    plt.imshow(img)

    mask_graph_skeleton = Image.open(os.path.join(pred_results, img_filename))
    mask_graph_skeleton = np.array(mask_graph_skeleton, dtype=np.uint8)

    indxs_skel = np.argwhere(mask_graph_skeleton==255)
    plt.scatter(indxs_skel[:,1],indxs_skel[:,0],color='red',marker='+')
    plt.axis('off')
    plt.show()


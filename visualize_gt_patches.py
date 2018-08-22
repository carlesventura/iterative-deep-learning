__author__ = 'carlesv'

from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

vessels = True
save_fig = False

if vessels:
    #img_id = 1
    #patch_id = 38

    #img_id_connected = 1
    #patch_id_connected = 23

    #img_id_same_vessel = 1
    #patch_id_same_vessel = 23


    #ids for CVPR figure
    img_id_connected = 12
    patch_id_connected = 45

    img_id_same_vessel = 1
    patch_id_same_vessel = 40

    db_root_dir_same_vessel = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected_same_vessel/'
    db_root_dir_connected = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected/'
    db_root_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_not_connected/'

else:
    db_root_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/val_gt/'
    img_id = 2
    patch_id = 46

if vessels:

    img = Image.open(os.path.join(db_root_dir, 'img_%02d_patch_%02d_img.png' %(img_id_connected,patch_id_connected)))
    gt = Image.open(os.path.join(db_root_dir, 'img_%02d_patch_%02d_gt.png' %(img_id_connected,patch_id_connected)))
    img = np.array(img,np.uint8)
    gt = np.array(gt,np.uint8)
    gt_points = np.argwhere(gt>200)
    plt.imshow(img)
    plt.scatter(gt_points[:,1],gt_points[:,0],color='green',marker='+', s=100, linewidth=5)
    plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
    plt.axis('off')
    if save_fig:
        #plt.savefig('/scratch_net/boxy/carlesv/retinal/CVPR2018/figures/train_patch_vessels_not_connected_2.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    gt_same_vessel = Image.open(os.path.join(db_root_dir_connected, 'img_%02d_patch_%02d_gt.png' %(img_id_connected,patch_id_connected)))
    gt_same_vessel = np.array(gt_same_vessel,np.uint8)
    gt_points = np.argwhere(gt_same_vessel==255)
    plt.imshow(img)
    plt.scatter(gt_points[:,1],gt_points[:,0],color='green',marker='+', s=100, linewidth=5)
    plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
    plt.axis('off')
    if save_fig:
        #plt.savefig('/scratch_net/boxy/carlesv/retinal/CVPR2018/figures/train_patch_vessels_connected_2.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    img = Image.open(os.path.join(db_root_dir_connected, 'img_%02d_patch_%02d_img.png' %(img_id_same_vessel,patch_id_same_vessel)))
    gt_connected = Image.open(os.path.join(db_root_dir_connected, 'img_%02d_patch_%02d_gt.png' %(img_id_same_vessel,patch_id_same_vessel)))
    img = np.array(img,np.uint8)
    gt_connected = np.array(gt_connected,np.uint8)
    gt_points = np.argwhere(gt_connected==255)
    plt.imshow(img)
    plt.scatter(gt_points[:,1],gt_points[:,0],color='green',marker='+', s=100, linewidth=5)
    plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
    plt.axis('off')
    if save_fig:
        #plt.savefig('/scratch_net/boxy/carlesv/retinal/CVPR2018/figures/train_patch_vessels_connected_different_vessels.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    gt_same_vessel = Image.open(os.path.join(db_root_dir_same_vessel, 'img_%02d_patch_%02d_gt.png' %(img_id_same_vessel,patch_id_same_vessel)))
    gt_same_vessel = np.array(gt_same_vessel,np.uint8)
    gt_points = np.argwhere(gt_same_vessel==255)
    plt.imshow(img)
    plt.scatter(gt_points[:,1],gt_points[:,0],color='green',marker='+', s=100, linewidth=5)
    plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
    plt.axis('off')
    if save_fig:
        #plt.savefig('/scratch_net/boxy/carlesv/retinal/CVPR2018/figures/train_patch_vessels_connected_same_vessels.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()
else:
    img = Image.open(os.path.join(db_root_dir, 'img_%02d_patch_%02d_img.png' %(img_id,patch_id)))
    gt = Image.open(os.path.join(db_root_dir, 'img_%02d_patch_%02d_gt.png' %(img_id,patch_id)))
    img = np.array(img,np.uint8)
    gt = np.array(gt,np.uint8)
    gt_points = np.argwhere(gt==255)
    plt.imshow(img)
    plt.scatter(gt_points[:,1],gt_points[:,0],color='red',marker='+', s=100, linewidth=5)
    plt.scatter(32,32,color='cyan',marker='+', s=100, linewidth=5)
    plt.axis('off')
    if save_fig:
        #plt.savefig('/scratch_net/boxy/carlesv/retinal/CVPR2018/figures/train_patch_roads_connected.png',bbox_inches='tight')
        plt.close()
    else:
        plt.show()




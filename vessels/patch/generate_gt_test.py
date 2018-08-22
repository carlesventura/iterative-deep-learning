
import numpy as np
import bifurcations_toolbox as tb
from torch.utils.data import DataLoader
import scipy.misc
import os

p = {}
p['inputRes'] = (64, 64)  # Input Resolution
p['outputRes'] = (64, 64)  # Output Resolution (same as input)
p['g_size'] = 64  # Higher means narrower Gaussian
p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images
db_root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
resultsOnTraining = False
junctions = True
connected = True
from_same_vessel = True
bifurcations_allowed = False
save_vertices_indxs = False
load_vertices_indxs = True

if junctions:
    save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_junctions/'
else:
    if connected:
        if from_same_vessel:
            if bifurcations_allowed:
                save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected_same_vessel/'
            else:
                save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected_same_vessel_wo_bifurcations/'
        else:
            save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_connected/'
    else:
        save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/gt_test_not_connected/'

composed_transforms_test = tb.ToTensor()

db_test = tb.ToolDataset(train=resultsOnTraining, inputRes=p['inputRes'], outputRes=p['outputRes'],
                         sigma=float(p['outputRes'][0]) / p['g_size'],
                         db_root_dir=db_root_dir, transform=composed_transforms_test, gt_masks=p['GTmasks'], junctions=junctions,
                         connected=connected, from_same_vessel=from_same_vessel, bifurcations_allowed=bifurcations_allowed, save_vertices_indxs=save_vertices_indxs)
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

num_patches_per_image = 50

if load_vertices_indxs:
    num_images = 20
    f = open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/vertices_selected_margin_6.txt','r')
    for jj in range(0,num_patches_per_image):
        for ii in range(0,num_images):
            line = f.readline()
            selected_vertex = int(line.split()[1])
            #ToDo: Generate img and gt pair from selected vertex (use functions already implemented in bifurcations_toolbox)


            if not os.path.isfile(save_dir + 'img_%02d_patch_%02d_img.png' %(ii+1,jj+1)):

                patch_size = p['inputRes'][0]
                input_on_left_rotated = False
                if junctions:
                    img_crop, output_points = tb.find_junctions_selected_vertex(db_root_dir, selected_vertex, False, ii+1, patch_size)
                else:
                    if connected:
                        img_crop, output_points = tb.find_output_connected_points_selected_vertex(db_root_dir, selected_vertex, False, ii+1, patch_size, from_same_vessel, bifurcations_allowed)
                    else:
                        img_crop, output_points = tb.find_output_points_selected_vertex(db_root_dir, selected_vertex, False, ii+1, patch_size, input_on_left_rotated)

                output_points = np.round(output_points) #round intersect points to generate gaussians all centered on pixels
                if len(output_points) > 1:
                    output_points = np.vstack({tuple(row) for row in output_points}) #remove duplicated values on output_points


                gt = tb.make_gt(img_crop, output_points, (patch_size,patch_size), float(p['outputRes'][0]) / p['g_size'])
                scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_gt.png' %(ii+1,jj+1), np.squeeze(gt))
                np.save(save_dir + 'img_%02d_patch_%02d_gt.npy' %(ii+1,jj+1), np.squeeze(gt))
                scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_img.png' %(ii+1,jj+1), img_crop)

else:
    for jj in range(0,num_patches_per_image):
        for ii, sample_batched in enumerate(testloader):
            img, gt = sample_batched['image'], sample_batched['gt']
            scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_gt.png' %(ii+1,jj+1), np.squeeze(gt.numpy()))
            np.save(save_dir + 'img_%02d_patch_%02d_gt.npy' %(ii+1,jj+1), np.squeeze(gt.numpy()))
            scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_img.png' %(ii+1,jj+1), np.transpose(np.squeeze(img.numpy()),(1,2,0)))

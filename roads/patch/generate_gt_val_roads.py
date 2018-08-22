import numpy as np
from torch.utils.data import DataLoader
import scipy.misc
import bifurcations_toolbox_roads as tbroads

# save_dir parent directory (results_dir) should match with save_dir parameter from train_road_patches.py
save_dir = './results_dir/gt_val/'
db_root_dir = './gt_dbs/MassachusettsRoads/'

p = {}
p['inputRes'] = (64, 64)  # Input Resolution
p['outputRes'] = (64, 64)  # Output Resolution (same as input)
p['g_size'] = 64  # Higher means narrower Gaussian
p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images

resultsOnTraining = False
save_vertices_indxs = True
load_vertices_indxs = False

composed_transforms_test = tbroads.ToTensor()

db_val = tbroads.ToolDataset(train=resultsOnTraining, inputRes=p['inputRes'], outputRes=p['outputRes'],
                         sigma=float(p['outputRes'][0]) / p['g_size'],
                         db_root_dir=db_root_dir, transform=composed_transforms_test, save_vertices_indxs=save_vertices_indxs)
valloader = DataLoader(db_val, batch_size=1, shuffle=False)

num_patches_per_image = 50

for jj in range(0,num_patches_per_image):
    for ii, sample_batched in enumerate(valloader):
        img, gt, valid_img = sample_batched['image'], sample_batched['gt'], sample_batched['valid_img']
        if int(valid_img.numpy()) == 1:
            scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_gt.png' %(ii+1,jj+1), np.squeeze(gt.numpy()))
            np.save(save_dir + 'img_%02d_patch_%02d_gt.npy' %(ii+1,jj+1), np.squeeze(gt.numpy()))
            scipy.misc.imsave(save_dir + 'img_%02d_patch_%02d_img.png' %(ii+1,jj+1), np.transpose(np.squeeze(img.numpy()),(1,2,0)))


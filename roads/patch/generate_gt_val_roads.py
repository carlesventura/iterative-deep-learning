import numpy as np
from torch.utils.data import DataLoader
import scipy.misc
import bifurcations_toolbox_roads as tbroads

p = {}
p['inputRes'] = (64, 64)  # Input Resolution
p['outputRes'] = (64, 64)  # Output Resolution (same as input)
p['g_size'] = 64  # Higher means narrower Gaussian
p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images
db_root_dir = '/scratch_net/boxy/carlesv/gt_dbs/MassachusettsRoads/'
resultsOnTraining = False
save_vertices_indxs = True
load_vertices_indxs = False

save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/roads/Iterative_margin_6/gt_val/'

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


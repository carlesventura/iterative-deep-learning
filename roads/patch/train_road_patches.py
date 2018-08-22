# Includes
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import Nets as nt
import timeit
import scipy.misc
from PIL import Image
import bifurcations_toolbox_roads as tbroads

# Setting of parameters
# db_root_dir and save_dir parameters should be changed by the user

db_root_dir = 'gt_dbs/MassachusettsRoads/' #Directory where the Massachusetts Roads Dataset is stored
save_dir = 'results_dir/' #Directory where the model will be stored during the training

# Parameters in p are used for the name of the model

p = {}
p['useRandom'] = 1  # Shuffle Images
p['useAug'] = 0  # Use Random rotations in [-30, 30] and scaling in [.75, 1.25]
p['inputRes'] = (64, 64)  # Input Resolution
p['outputRes'] = (64, 64)  # Output Resolution (same as input)
p['g_size'] = 64  # Higher means narrower Gaussian
p['trainBatch'] = 1  # Number of Images in each mini-batch
p['numHG'] = 2  # Number of Stacked Hourglasses
p['Block'] = 'ConvBlock'  # Select: 'ConvBlock', 'BasicBlock', 'BottleNeck', 'BottleneckPreact'
p['GTmasks'] = 0 # Use GT Vessel Segmentations as input instead of Retinal Images

save_vertices_indxs = False

# Setting other parameters
nEpochs = 200  # Number of epochs for training
numHGScales = 4  # How many times to downsample inside each HourGlass
if 'SGE_GPU' in os.environ.keys():
    gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
    print('GPU:'+str(gpu_id))
else:
    gpu_id = -1
snapshot = 10  # Store a model every snapshot epochs


# Network definition
net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

# Loss function definition
criterion = nn.MSELoss(size_average=True)

# Use the following optimizer
optimizer = optim.RMSprop(net.parameters(), lr=0.00005, alpha=0.99, momentum=0.0)


# Preparation of the data loaders
# Define augmentation transformations as a composition
if p['useAug'] == 1:
    composed_transforms = transforms.Compose([tbroads.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)), tbroads.ToTensor()])
else:
    composed_transforms = tbroads.ToTensor()


# Training dataset and its iterator
db_train = tbroads.ToolDataset(train=True, inputRes=p['inputRes'], outputRes=p['outputRes'],
                          sigma=float(p['outputRes'][0]) / p['g_size'],
                          db_root_dir=db_root_dir, transform=composed_transforms, save_vertices_indxs=False)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True)

num_img_tr = len(trainloader)

running_loss_tr = 0
loss_tr = []


modelName = tbroads.construct_name(p, "HourGlass")


print("Training Network")

file = open(save_dir + 'logfile.txt', 'a')
file_training_loss = open(save_dir + 'training_loss.txt', 'a')

# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    start_time = timeit.default_timer()
    file.write('Epoch %02d\n' % (epoch))
    file.flush()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        img, gt, valid_img = sample_batched['image'], sample_batched['gt'], sample_batched['valid_img']
        file.write('Image %02d\n' % (ii))
        file.flush()

        if int(valid_img.numpy()) == 1:
            inputs = img / 255 - 0.5
            gts = 255 * gt
            file.write('Image %02d being processed\n' % (ii))
            file.flush()

            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs), Variable(gts)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            optimizer.zero_grad()
            outputs = net.forward(inputs)

            losses = [None] * p['numHG']
            for i in range(0, len(outputs)):
                losses[i] = criterion(outputs[i], gts)
            loss = sum(losses)

            running_loss_tr += loss.data[0]
            if ii % num_img_tr == num_img_tr-1:
                loss_tr.append(running_loss_tr/num_img_tr)
                print('[%d, %5d] training loss: %.5f' % (epoch+1, ii + 1, running_loss_tr/num_img_tr))
                file_training_loss.write('[%d, %5d] training loss: %.5f' % (epoch+1, ii + 1, running_loss_tr/num_img_tr))
                file_training_loss.flush()
                running_loss_tr = 0
            loss.backward()
            optimizer.step()


    # Save the model
    if (epoch % snapshot) == 0 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(save_dir, modelName+'_epoch-'+str(epoch)+'.pth'))



file.close()
file_training_loss.close()


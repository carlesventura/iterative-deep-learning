# Includes
import os
import numpy as np
import bifurcations_toolbox as tb
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import Nets as nt
import timeit
import scipy.misc

# Setting of parameters
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

junctions = True
connected = True
from_same_vessel = True
bifurcations_allowed = False
save_vertices_indxs = False

# Setting other parameters
nEpochs = 2000  # Number of epochs for training
numHGScales = 4  # How many times to downsample inside each HourGlass
useTest = 1  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 10  # Run on test set every nTestInterval iterations
#seq_names = ('1-left', '3-left', '2-left')
db_root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
save_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/'
vis_net = 0  # Visualize your network?
gpu_id = int(os.environ['SGE_GPU'])  # Select which GPU, -1 if CPU
#gpu_id = -1
snapshot = 200  # Store a model every snapshot epochs


# Network definition
net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

# Loss function definition
criterion = nn.MSELoss(size_average=True)

# Use the following optimizer
optimizer = optim.RMSprop(net.parameters(), lr=0.00005, alpha=0.99, momentum=0.0)
# optimizer = optim.SGD(net.parameters(), lr=0.01/1000, momentum=0.1)
# optimizer = optim.Adam(net.parameters(), lr=0.0005)


# Preparation of the data loaders
# Define augmentation transformations as a composition
#composed_transforms = transforms.Compose([tb.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)), tb.RandomHorizontalFlip(), tb.ToTensor()])
if p['useAug'] == 1:
    #composed_transforms = transforms.Compose([tb.ScaleNRotate(rots=(-90, 90), scales=(.5, 1.5)), tb.ToTensor(), tb.normalize()])
    composed_transforms = transforms.Compose([tb.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)), tb.ToTensor()])
else:
    composed_transforms = tb.ToTensor()

composed_transforms_test = tb.ToTensor()

# Training dataset and its iterator
db_train = tb.ToolDataset(train=True, inputRes=p['inputRes'], outputRes=p['outputRes'],
                          sigma=float(p['outputRes'][0]) / p['g_size'],
                          db_root_dir=db_root_dir, transform=composed_transforms, gt_masks=p['GTmasks'], junctions=junctions,
                          connected=connected, from_same_vessel=from_same_vessel,bifurcations_allowed=bifurcations_allowed,save_vertices_indxs=save_vertices_indxs)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True)

# Testing dataset and its iterator
db_test = tb.ToolDataset(train=False, inputRes=p['inputRes'], outputRes=p['outputRes'],
                         sigma=float(p['outputRes'][0]) / p['g_size'],
                         db_root_dir=db_root_dir, transform=composed_transforms_test, gt_masks=p['GTmasks'], junctions=junctions,
                         connected=connected, from_same_vessel=from_same_vessel,bifurcations_allowed=bifurcations_allowed,save_vertices_indxs=save_vertices_indxs)
testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False)


num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = 0
running_loss_ts = 0
loss_tr = []
loss_ts = []

if junctions:
    modelName = tb.construct_name(p, "HourGlass-junctions")
else:
    if not connected:
        modelName = tb.construct_name(p, "HourGlass")
    else:
        if from_same_vessel:
            if bifurcations_allowed:
                modelName = tb.construct_name(p, "HourGlass-connected-same-vessel")
            else:
                modelName = tb.construct_name(p, "HourGlass-connected-same-vessel-wo-bifurcations")
        else:
            modelName = tb.construct_name(p, "HourGlass-connected")

print("Training Network")

# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        img, gt = sample_batched['image'], sample_batched['gt']
        inputs = img / 255 - 0.5
        gts = 255 * gt


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
            running_loss_tr = 0
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))
        loss.backward()
        optimizer.step()

    # Save the model
    if (epoch % snapshot) == 0 and epoch != 0:
        torch.save(net.state_dict(), os.path.join(save_dir, modelName+'_epoch-'+str(epoch)+'.pth'))

    # One testing epoch
    if useTest:
        if epoch % nTestInterval == (nTestInterval-1):
            for ii, sample_batched in enumerate(testloader):
                img, gt = sample_batched['image'], sample_batched['gt']
                inputs = img / 255 - 0.5
                gts = 255 * gt

                # Forward pass of the mini-batch
                inputs, gts = Variable(inputs), Variable(gts)
                if gpu_id >= 0:
                    inputs, gts = inputs.cuda(), gts.cuda()

                output = net.forward(inputs)

                if ii == 0:
                    pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))

                    if junctions:
                        base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/tmp_results_junctions/'
                    else:
                        if not connected:
                            base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/tmp_results_not_connected/'
                        else:
                            if from_same_vessel:
                                if bifurcations_allowed:
                                    base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/tmp_results_connected_same_vessel/'
                                else:
                                    base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/tmp_results_connected_same_vessel_wo_bifurcations/'
                            else:
                                base_dir = '/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/tmp_results_connected/'

                    scipy.misc.imsave(base_dir + 'epoch_' + str(epoch) + '_test.png', pred)
                    scipy.misc.imsave(base_dir + 'epoch_' + str(epoch) + '_gt.png', np.squeeze(gt.numpy()))
                    np.save(base_dir + 'epoch_' + str(epoch) + '_gt.npy', np.squeeze(gt.numpy()))
                    scipy.misc.imsave(base_dir + 'epoch_' + str(epoch) + '_img.png', np.transpose(np.squeeze(img.numpy()),(1,2,0)))

                losses_test = [None] * p['numHG']
                for i in range(0, len(outputs)):
                    losses_test[i] = criterion(outputs[i], gts)
                loss = sum(losses_test)

                running_loss_ts += loss.data[0]
                if ii % num_img_ts == num_img_ts-1:
                    loss_ts.append(running_loss_ts / num_img_ts)
                    print('[%d, %5d] ***** testing ***** loss: %.5f' % (epoch+1, ii + 1, running_loss_ts/num_img_ts))
                    running_loss_ts = 0


# Separate interactive plotting of test set results
# plt.close("all")
# plt.ion()
# f, ax_arr = plt.subplots(1, 3)
# for ii, sample_batched in enumerate(testloader):
#     img, gt = sample_batched['image'], sample_batched['gt']
#     inputs = img / 255 - 0.5
#
#     # Forward pass of the mini-batch
#     inputs = Variable(inputs)
#     if gpu_id >= 0:
#         inputs = inputs.cuda()
#
#     output = net.forward(inputs)
#     for jj in range(int(img.size()[0])):
#         pred = np.transpose(output[len(output)-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
#         img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
#         gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
#         # Plot the particular example
#         ax_arr[0].cla()
#         ax_arr[1].cla()
#         ax_arr[2].cla()
#         ax_arr[0].set_title("Input Image")
#         ax_arr[1].set_title("Ground Truth")
#         ax_arr[2].set_title("Detection")
#         ax_arr[0].imshow(tb.im_normalize(img_))
#         ax_arr[1].imshow(gt_)
#         ax_arr[2].imshow(tb.im_normalize(pred))
#         plt.pause(0.001)


# Separate interactive plotting of training set results
# plt.close("all")
# plt.ion()
# f, ax_arr = plt.subplots(1, 3)
# for ii, sample_batched in enumerate(trainloader):
#     img, gt = sample_batched['image'], sample_batched['gt']
#     inputs = img / 255 - 0.5
#
#     # Forward pass of the mini-batch
#     inputs = Variable(inputs)
#     if gpu_id >= 0:
#         inputs = inputs.cuda()
#
#     output = net.forward(inputs)
#     for jj in range(int(img.size()[0])):
#         pred = np.transpose(output[len(output)-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
#         img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
#         gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
#         # Plot the particular example
#         ax_arr[0].cla()
#         ax_arr[1].cla()
#         ax_arr[2].cla()
#         ax_arr[0].set_title("Input Image")
#         ax_arr[1].set_title("Ground Truth")
#         ax_arr[2].set_title("Detection")
#         ax_arr[0].imshow(tb.im_normalize(img_))
#         ax_arr[1].imshow(gt_)
#         ax_arr[2].imshow(tb.im_normalize(pred))
#         plt.pause(0.001)

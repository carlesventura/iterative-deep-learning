from __future__ import division
import numpy as np
from PIL import Image
import cv2
import scipy.io as sio
import os
import random
import torch
from torch.utils.data import Dataset
import intersection_graph_bbox as igb


def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / (im.max() - im.min())
    return imn


def construct_name(p, prefix):
    """
    Construct the name of the model
    p: dictionary of parameters
    prefix: the prefix
    name: the name of the model - manually add ".pth" to follow the convention
    """
    name = prefix
    for key in p.keys():
        if (type(p[key]) != tuple) and (type(p[key]) != list):
            name = name + '_' + str(key) + '-' + str(p[key])
        else:
            name = name + '_' + str(key) + '-' + str(p[key][0])
    return name


def make_gaussian(size, sigma=10, center=None):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def make_gt(img, labels, outputRes=None, sigma=10):
    """ Make the ground-truth for each landmark.
    img: the original color image
    labels: the json labels with the Gaussian centers {'x': x, 'y': y}
    sigma: sigma of the Gaussian.
    """

    if outputRes is not None:
        h, w = outputRes
    else:
        h, w = img.shape
    # print (h, w, len(labels))
    #gt = np.zeros((h, w, len(labels)), np.float32)
    gt = np.zeros((h, w, 1), np.float32)

    for land in range(0, labels.shape[0]):
        gt[:,:,0] = gt[:,:,0] + (make_gaussian((h, w), sigma, (labels[land, 0], labels[land, 1])))
    return gt


def txt2mat(idx):
    annotation_file = '/scratch_net/boxy/carlesv/RetinalFeatures/%02d_manual1_gt.txt' % (idx)
    file = open(annotation_file, 'r')
    lines = file.readlines()
    file.close()
    x = np.nan * np.zeros((len(lines), 2))
    for i in range(0,len(lines)):
        x[i, 0] = int(lines[i].split(',')[1])
        x[i, 1] = int(lines[i].split(',')[0])
    return x


def overlay_mask(img, mask, transparency=0.5):
    """
    Overlay a h x w x 3 mask to the image
    img: h x w x 3 image
    mask: h x w x 3 mask
    transparency: between 0 and 1
    """
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask[:, :, 0]) * img[:, :, 0] + mask[:, :, 0] * (
    255 * transparency + (1 - transparency) * img[:, :, 0])
    im_over[:, :, 1] = (1 - mask[:, :, 1]) * img[:, :, 1] + mask[:, :, 1] * (
    255 * transparency + (1 - transparency) * img[:, :, 1])
    im_over[:, :, 2] = (1 - mask[:, :, 2]) * img[:, :, 2] + mask[:, :, 2] * (
    255 * transparency + (1 - transparency) * img[:, :, 2])
    return im_over

def find_output_points(root_dir, save_vertices_indxs, train, img_idx, patch_size, input_on_left_rotated):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    margin = int(np.round(patch_size/10.0))

    #Select a random vertex from the graph
    selected_vertex = random.randint(0,len(vertices)-1)
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])
    while center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2:
        selected_vertex = random.randint(0,len(vertices)-1)
        center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    #Add selected vertex to file to reproduce the experiments
    if save_vertices_indxs:
        f = open(os.path.join(root_dir, 'vertices_selected.txt'), 'a')
        f.write(str(img_idx) + " " + str(selected_vertex) + "\n")
        f.close()

    if input_on_left_rotated:
        #Find previous connected vertex
        edges = np.array(graph['E'])
        #print(edges[0,0][424,1023])
        subset_edges = edges[0,0][selected_vertex,:]
        subset_vertices = np.argwhere(subset_edges)
        subset_vertices = subset_vertices[:,1]
        previous_vertex = subset_vertices[random.randint(0,len(subset_vertices)-1)]

        rot = np.arctan2(vertices[selected_vertex,1]-vertices[previous_vertex,1],vertices[selected_vertex,0]-vertices[previous_vertex,0])*180/np.pi
        sc = 1
        M = cv2.getRotationMatrix2D(center, rot, sc)

        img_ = cv2.warpAffine(img, M, (w, h))
        x_tmp = int(center[0]-margin)
        y_tmp = int(center[1]-patch_size/2)
        img_crop = img_[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

        bbox = np.array([[center[0],center[0]+patch_size-2*margin,center[0]+patch_size-2*margin,center[0]],
                         [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin,center[1]+patch_size/2-margin]])

        bbox = np.reshape(bbox,(2,4,1))

        M_inv = cv2.getRotationMatrix2D(center, -rot, sc)

        bbox_rotated = cv2.transform(np.transpose(bbox),M_inv)

        bbox_rotated = np.reshape(bbox_rotated,(4,2))

        intersect_points = igb.intersect(train,img_idx, bbox_rotated)

        #remove center (input vertex) from intersect points
        output_points = []
        for ii in range(0,len(intersect_points)):
            tmp_vector = intersect_points[ii] - center
            norm = tmp_vector[0]*tmp_vector[0]+tmp_vector[1]*tmp_vector[1]
            if norm > 1:
                output_points.append(intersect_points[ii])

        #generate ground truth output (generate the gaussian and perform the rotation or viceversa)
        output_points_tmp = np.zeros((2,len(output_points),1))
        for ii in range(0,len(output_points)):
            output_points_tmp[0,ii,0]=output_points[ii][0]
            output_points_tmp[1,ii,0]=output_points[ii][1]

        output_points_rotated = cv2.transform(np.transpose(output_points_tmp),M)
        xy = [x_tmp, y_tmp]
        output_points_rotated = output_points_rotated - xy
        output_points_rotated = output_points_rotated.astype(int)
        output_points = np.squeeze(output_points_rotated)

    else:
        x_tmp = int(center[0]-patch_size/2)
        y_tmp = int(center[1]-patch_size/2)
        img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]
        bbox = np.array([[center[0]-patch_size/2+margin,center[0]+patch_size/2-margin-1,center[0]+patch_size/2-margin-1,center[0]-patch_size/2+margin],
                         [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin-1,center[1]+patch_size/2-margin-1]])
        bbox = np.transpose(bbox)
        intersect_points = igb.intersect(train,img_idx,bbox)
        output_points = np.asarray(intersect_points)
        xy = [x_tmp, y_tmp]
        output_points = output_points - xy

        # plt.figure()
        # plt.imshow(Image.fromarray(np.uint8(img_crop)))
        # plt.plot(output_points[:,0], output_points[:,1], ls='none', color='blue',marker='+', ms=10, lw=1.5)
        #
        # subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
        # for ii in range(0,len(subscripts)):
        #     segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
        #     xcoords, ycoords = segment.xy
        #     plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='green', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
        #
        # plt.xlim([0, patch_size-1])
        # plt.ylim([patch_size-1,0])
        # plt.show(block=False)

    return img_crop, output_points

def find_output_points_selected_vertex(root_dir, selected_vertex, train, img_idx, patch_size, input_on_left_rotated):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    margin = int(np.round(patch_size/10.0))

    #Use vertex from input parameter selected_vertex
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    if input_on_left_rotated:
        #Find previous connected vertex
        edges = np.array(graph['E'])
        #print(edges[0,0][424,1023])
        subset_edges = edges[0,0][selected_vertex,:]
        subset_vertices = np.argwhere(subset_edges)
        subset_vertices = subset_vertices[:,1]
        previous_vertex = subset_vertices[random.randint(0,len(subset_vertices)-1)]

        rot = np.arctan2(vertices[selected_vertex,1]-vertices[previous_vertex,1],vertices[selected_vertex,0]-vertices[previous_vertex,0])*180/np.pi
        sc = 1
        M = cv2.getRotationMatrix2D(center, rot, sc)

        img_ = cv2.warpAffine(img, M, (w, h))
        x_tmp = int(center[0]-margin)
        y_tmp = int(center[1]-patch_size/2)
        img_crop = img_[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

        bbox = np.array([[center[0],center[0]+patch_size-2*margin,center[0]+patch_size-2*margin,center[0]],
                         [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin,center[1]+patch_size/2-margin]])

        bbox = np.reshape(bbox,(2,4,1))

        M_inv = cv2.getRotationMatrix2D(center, -rot, sc)

        bbox_rotated = cv2.transform(np.transpose(bbox),M_inv)

        bbox_rotated = np.reshape(bbox_rotated,(4,2))

        intersect_points = igb.intersect(train,img_idx, bbox_rotated)

        #remove center (input vertex) from intersect points
        output_points = []
        for ii in range(0,len(intersect_points)):
            tmp_vector = intersect_points[ii] - center
            norm = tmp_vector[0]*tmp_vector[0]+tmp_vector[1]*tmp_vector[1]
            if norm > 1:
                output_points.append(intersect_points[ii])

        #generate ground truth output (generate the gaussian and perform the rotation or viceversa)
        output_points_tmp = np.zeros((2,len(output_points),1))
        for ii in range(0,len(output_points)):
            output_points_tmp[0,ii,0]=output_points[ii][0]
            output_points_tmp[1,ii,0]=output_points[ii][1]

        output_points_rotated = cv2.transform(np.transpose(output_points_tmp),M)
        xy = [x_tmp, y_tmp]
        output_points_rotated = output_points_rotated - xy
        output_points_rotated = output_points_rotated.astype(int)
        output_points = np.squeeze(output_points_rotated)

    else:
        x_tmp = int(center[0]-patch_size/2)
        y_tmp = int(center[1]-patch_size/2)
        img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]
        bbox = np.array([[center[0]-patch_size/2+margin,center[0]+patch_size/2-margin-1,center[0]+patch_size/2-margin-1,center[0]-patch_size/2+margin],
                         [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin-1,center[1]+patch_size/2-margin-1]])
        bbox = np.transpose(bbox)
        intersect_points = igb.intersect(train,img_idx,bbox)
        output_points = np.asarray(intersect_points)
        xy = [x_tmp, y_tmp]
        output_points = output_points - xy

        # plt.figure()
        # plt.imshow(Image.fromarray(np.uint8(img_crop)))
        # plt.plot(output_points[:,0], output_points[:,1], ls='none', color='blue',marker='+', ms=10, lw=1.5)
        #
        # subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
        # for ii in range(0,len(subscripts)):
        #     segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
        #     xcoords, ycoords = segment.xy
        #     plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='green', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
        #
        # plt.xlim([0, patch_size-1])
        # plt.ylim([patch_size-1,0])
        # plt.show(block=False)

    return img_crop, output_points

def find_output_connected_points(root_dir, save_vertices_indxs, train, img_idx, patch_size, from_same_vessel, bifurcations_allowed):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    margin = int(np.round(patch_size/10.0))


    #Select a random vertex from the graph
    selected_vertex = random.randint(0,len(vertices)-1)
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])
    while center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2:
        selected_vertex = random.randint(0,len(vertices)-1)
        center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    #Add selected vertex to file to reproduce the experiments
    if save_vertices_indxs:
        f = open(os.path.join(root_dir, 'vertices_selected.txt'), 'a')
        f.write(str(img_idx) + " " + str(selected_vertex) + "\n")
        f.close()

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]
    bbox = np.array([[center[0]-patch_size/2+margin,center[0]+patch_size/2-margin-1,center[0]+patch_size/2-margin-1,center[0]-patch_size/2+margin],
                     [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin-1,center[1]+patch_size/2-margin-1]])
    bbox = np.transpose(bbox)
    if from_same_vessel:
        art = np.squeeze(mat_contents['G']['art'][0,0])
        ven = np.squeeze(mat_contents['G']['ven'][0,0])
        if art[selected_vertex] == 0 and ven[selected_vertex] == 0:
            artery = 2 #unknown
        else:
            artery = art[selected_vertex]
        intersect_points = igb.intersect_connected_same_vessel(train,img_idx,bbox,selected_vertex,artery,bifurcations_allowed)
    else:
        intersect_points = igb.intersect_connected(train,img_idx,bbox,selected_vertex)
    output_points = np.asarray(intersect_points)
    xy = [x_tmp, y_tmp]
    if output_points.shape[0] > 0:
        output_points = output_points - xy

    # plt.figure()
    # plt.imshow(Image.fromarray(np.uint8(img_crop)))
    # plt.plot(output_points[:,0], output_points[:,1], ls='none', color='blue',marker='+', ms=10, lw=1.5)
    #
    # subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
    # for ii in range(0,len(subscripts)):
    #     segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
    #     xcoords, ycoords = segment.xy
    #     plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='green', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
    #
    # plt.xlim([0, patch_size-1])
    # plt.ylim([patch_size-1,0])
    # plt.show(block=False)

    return img_crop, output_points

def find_output_connected_points_selected_vertex(root_dir, selected_vertex, train, img_idx, patch_size, from_same_vessel, bifurcations_allowed):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    margin = int(np.round(patch_size/10.0))


    #Use vertex from input parameter selected_vertex
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]
    bbox = np.array([[center[0]-patch_size/2+margin,center[0]+patch_size/2-margin-1,center[0]+patch_size/2-margin-1,center[0]-patch_size/2+margin],
                     [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin-1,center[1]+patch_size/2-margin-1]])
    bbox = np.transpose(bbox)
    if from_same_vessel:
        art = np.squeeze(mat_contents['G']['art'][0,0])
        ven = np.squeeze(mat_contents['G']['ven'][0,0])
        if art[selected_vertex] == 0 and ven[selected_vertex] == 0:
            artery = 2 #unknown
        else:
            artery = art[selected_vertex]
        intersect_points = igb.intersect_connected_same_vessel(train,img_idx,bbox,selected_vertex,artery,bifurcations_allowed)
    else:
        intersect_points = igb.intersect_connected(train,img_idx,bbox,selected_vertex)
    output_points = np.asarray(intersect_points)
    xy = [x_tmp, y_tmp]
    if output_points.shape[0] > 0:
        output_points = output_points - xy

    # plt.figure()
    # plt.imshow(Image.fromarray(np.uint8(img_crop)))
    # plt.plot(output_points[:,0], output_points[:,1], ls='none', color='blue',marker='+', ms=10, lw=1.5)
    #
    # subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
    # for ii in range(0,len(subscripts)):
    #     segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
    #     xcoords, ycoords = segment.xy
    #     plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='green', alpha=0.5, linewidth=1, solid_capstyle='round', zorder=2)
    #
    # plt.xlim([0, patch_size-1])
    # plt.ylim([patch_size-1,0])
    # plt.show(block=False)

    return img_crop, output_points


def find_junctions(root_dir, save_vertices_indxs, train, img_idx, patch_size):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    #Select a random vertex from the graph
    selected_vertex = random.randint(0,len(vertices)-1)
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])
    while center[0] < patch_size/2 or center[1] < patch_size/2 or center[0] >  w - patch_size/2 or center[1] >  h - patch_size/2:
        selected_vertex = random.randint(0,len(vertices)-1)
        center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    #Add selected vertex to file to reproduce the experiments
    if save_vertices_indxs:
        f = open(os.path.join(root_dir, 'vertices_selected.txt'), 'a')
        f.write(str(img_idx) + " " + str(selected_vertex) + "\n")
        f.close()

    #Use vertex from input parameter selected_vertex
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    junction_points = igb.find_junctions(train,img_idx,selected_vertex)
    junction_points = np.asarray(junction_points)
    xy = [x_tmp, y_tmp]
    if junction_points.shape[0] > 0:
        junction_points = junction_points - xy

    return img_crop, junction_points


def find_junctions_selected_vertex(root_dir, selected_vertex, train, img_idx, patch_size):

    if train:
        img = Image.open(os.path.join(root_dir, 'training', 'images', '%02d_training.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %(img_idx))
    else:
        img = Image.open(os.path.join(root_dir, 'test', 'images', '%02d_test.tif' %(img_idx)))
        mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %(img_idx))

    img = np.array(img, dtype=np.float32)
    h, w = img.shape[:2]

    graph = mat_contents['G']

    vertices = np.squeeze(graph['V'][0,0])-1

    #Use vertex from input parameter selected_vertex
    center = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    x_tmp = int(center[0]-patch_size/2)
    y_tmp = int(center[1]-patch_size/2)
    img_crop = img[y_tmp:y_tmp+patch_size,x_tmp:x_tmp+patch_size,:]

    junction_points = igb.find_junctions(train,img_idx,selected_vertex)
    junction_points = np.asarray(junction_points)
    xy = [x_tmp, y_tmp]
    if junction_points.shape[0] > 0:
        junction_points = junction_points - xy

    return img_crop, junction_points


class ToolDataset(Dataset):
    """Tool Dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=(64, 64),
                 outputRes=(64, 64),
                 sigma=5,
                 db_root_dir='/scratch_net/boxy/carlesv/gt_dbs/DRIVE/',
                 transform=None,
                 gt_masks=0,
                 junctions=False,
                 connected=False,
                 from_same_vessel=False,
                 bifurcations_allowed=True,
                 save_vertices_indxs=False):
        """Loads image to label pairs for tool pose estimation
        db_elements: the names of the video files
        db_root_dir: dataset directory with subfolders "frames" and "Annotations"
        """
        self.train = train
        self.db_root_dir = db_root_dir
        self.inputRes = inputRes
        self.outputRes = outputRes
        self.sigma = sigma
        self.transform = transform
        self.gt_masks = gt_masks
        self.junctions = junctions
        self.connected = connected
        self.from_same_vessel = from_same_vessel
        self.bifurcations_allowed = bifurcations_allowed
        self.save_vertices_indxs = save_vertices_indxs

        if self.train:
            self.img_list = range(21,41)
        else:
            self.img_list = range(1,21)

        print('Done initializing Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)

        sample = {'image': img, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        if self.train:
            if self.gt_masks == 0:

                img_idx = self.img_list[idx]
                patch_size = self.inputRes[0]
                input_on_left_rotated = False

                if self.junctions:

                    img_crop, output_points = find_junctions(self.db_root_dir, False, self.train, img_idx, patch_size)

                else:

                    if self.connected:
                        img_crop, output_points = find_output_connected_points(self.db_root_dir, False, self.train, img_idx, patch_size, self.from_same_vessel, self.bifurcations_allowed)
                    else:
                        img_crop, output_points = find_output_points(self.db_root_dir, False, self.train, img_idx, patch_size, input_on_left_rotated)




                output_points = np.round(output_points) #round intersect points to generate gaussians all centered on pixels
                if len(output_points) > 1:
                    output_points = np.vstack({tuple(row) for row in output_points}) #remove duplicated values on output_points
                #print(output_points)

                gt = make_gt(img_crop, output_points, (patch_size,patch_size), self.sigma)
                #scipy.misc.imsave('/scratch_net/boxy/carlesv/HourGlasses_experiments/tmp_image.png', img_crop)
                #scipy.misc.imsave('/scratch_net/boxy/carlesv/HourGlasses_experiments/tmp_gt.png', np.squeeze(gt))

                return img_crop, gt
            else:
                img = Image.open(os.path.join(self.db_root_dir, 'training', '1st_manual', '%02d_manual1.gif' %(self.img_list[idx])))
                return img
        else:
            if self.gt_masks == 0:
                img_idx = self.img_list[idx]
                patch_size = self.inputRes[0]
                input_on_left_rotated = False
                if self.junctions:
                    img_crop, output_points = find_junctions(self.db_root_dir, self.save_vertices_indxs, self.train, img_idx, patch_size)
                else:
                    if self.connected:
                        img_crop, output_points = find_output_connected_points(self.db_root_dir, self.save_vertices_indxs, self.train, img_idx, patch_size, self.from_same_vessel, self.bifurcations_allowed)
                    else:
                        img_crop, output_points = find_output_points(self.db_root_dir, self.save_vertices_indxs, self.train, img_idx, patch_size, input_on_left_rotated)


                output_points = np.round(output_points) #round intersect points to generate gaussians all centered on pixels
                if len(output_points) > 1:
                    output_points = np.vstack({tuple(row) for row in output_points}) #remove duplicated values on output_points

                gt = make_gt(img_crop, output_points, (patch_size,patch_size), self.sigma)
                return img_crop, gt
            else:
                img = Image.open(os.path.join(self.db_root_dir, 'test', '1st_manual', '%02d_manual1.gif' %(self.img_list[idx])))
                return img


    def store_gt_asmatfile(self):
        gt_tool = np.zeros((2, 3, len(self.img_list)), dtype=np.float32)
        for i in range(0, len(self.img_list)):
            temp = txt2mat(self.img_list[i])
            if temp.shape[0] == 0:
                gt_tool[:, :, i] = np.nan
            else:
                gt_tool[:, :, i] = np.transpose(temp)

        a = {'gt_tool': gt_tool}
        sio.savemat('gt_tool', a)

    def get_img_size(self):
        if self.train:
            img = Image.open(os.path.join(self.db_root_dir, 'training', 'images', '%02d_training.tif' %(self.img_list[0])))
        else:
            img = Image.open(os.path.join(self.db_root_dir, 'test', 'images', '%02d_test.tif' %(self.img_list[0])))

        return list(reversed(img.size))


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        maxRot (float): maximum rotation angle to be added
        maxScale (float): maximum scale to be added
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        rot = (self.rots[1] - self.rots[0]) * random.random() - \
              (self.rots[1] - self.rots[0])/2

        sc = (self.scales[1] - self.scales[0]) * random.random() - \
             (self.scales[1] - self.scales[0]) / 2 + 1

        img, gt = sample['image'], sample['gt']
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, rot, sc)
        img_ = cv2.warpAffine(img, M, (w, h))

        h_gt, w_gt = gt.shape[:2]
        center_gt = (w_gt / 2, h_gt / 2)
        M = cv2.getRotationMatrix2D(center_gt, rot, sc)
        gt_ = cv2.warpAffine(gt, M, (w_gt, h_gt))

        return {'image': img_, 'gt': gt_}

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        image, gt = sample['image'], sample['gt']

        if random.random() < 0.5:
            image = cv2.flip(image, flipCode=1)
            gt = cv2.flip(gt, flipCode=1)

        sample['image'], sample['gt'] = image, gt

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(gt.shape) == 2:
            gt_tmp = gt
            h, w = gt_tmp.shape
            gt = np.zeros((h, w, 1), np.float32)
            gt[:,:,0] = gt_tmp
        if len(image.shape) == 2:
            image_tmp = image
            h, w = image_tmp.shape
            image = np.zeros((h, w, 3), np.float32)
            image[:,:,0] = image_tmp
            image[:,:,1] = image_tmp
            image[:,:,2] = image_tmp
        image = image.transpose((2, 0, 1))
        gt = gt.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt)}


class normalize(object):

    def __init__(self, mean=[171.0773/255, 98.4333/255, 58.8811/255], std=[1.0, 1.0, 1.0]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        image, gt = sample['image'], sample['gt']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)

        sample['image'], sample['gt'] = image, gt

        return sample


if __name__ == 'main':
    a = ToolDataset(train=True)

    for i, sample in enumerate(a):
        print(i)
        gt = sample['gt']
        if gt.shape[2] == 0:
            break

    b = a[153]
    #plt.imshow(b['image'])
    #plt.imshow(b['gt'])

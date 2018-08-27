__author__ = 'carlesv'

import scipy.io as sio
import numpy as np
from shapely.geometry import LineString


def intersect(train,idx,bbox):

    if train:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %idx)
    else:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])

    graph_segments = []

    for ii in range(0,len(subscripts)):
        segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
        graph_segments.append(segment)

    bbox_segments = LineString([bbox[0],bbox[1],bbox[2],bbox[3],bbox[0]])

    intersect_points = []

    for ii in range(0,len(subscripts)):
        points = graph_segments[ii].intersection(bbox_segments)
        if points.geom_type == 'Point':
            intersect_points.append(np.array(points))
        elif points.geom_type == 'MultiPoint':
            for jj in range(0,len(points)):
                intersect_points.append(np.array(points[jj]))

    return intersect_points

def find_next_point(selected_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points):

    previous_point = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    next_idxs = np.argwhere(subscripts==(selected_vertex+1))

    for ii in range(0,len(next_idxs)):

        if next_idxs[ii,1] == 0:
            next_vertex = subscripts[next_idxs[ii,0],1]-1
        else:
            next_vertex = subscripts[next_idxs[ii,0],0]-1

        if next_vertex not in visited_points:

            visited_points.append(next_vertex)
            next_point = (vertices[next_vertex,0], vertices[next_vertex,1])
            segment = LineString([previous_point, next_point])
            points = segment.intersection(bbox_segments)
            if points.geom_type == 'Point':
                intersect_points.append(np.array(points))
            else:
                find_next_point(next_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)

def find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, selected_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points):

    previous_point = (vertices[selected_vertex,0], vertices[selected_vertex,1])

    next_idxs = np.argwhere(subscripts==(selected_vertex+1))

    if bifurcations_allowed:

        for ii in range(0,len(next_idxs)):

            if next_idxs[ii,1] == 0:
                next_vertex = subscripts[next_idxs[ii,0],1]-1
            else:
                next_vertex = subscripts[next_idxs[ii,0],0]-1

            if next_vertex not in visited_points:

                visited_points.append(next_vertex)

                if artery == 2:
                    if art[next_vertex] != 0 or ven[next_vertex] != 0:
                        artery = art[next_vertex]

                    next_point = (vertices[next_vertex,0], vertices[next_vertex,1])
                    segment = LineString([previous_point, next_point])
                    points = segment.intersection(bbox_segments)
                    if points.geom_type == 'Point':
                        intersect_points.append(np.array(points))
                    else:
                        find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, next_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)

                else:
                    if (artery and art[next_vertex]) or ((not artery) and ven[next_vertex]):
                        next_point = (vertices[next_vertex,0], vertices[next_vertex,1])
                        segment = LineString([previous_point, next_point])
                        points = segment.intersection(bbox_segments)
                        if points.geom_type == 'Point':
                            intersect_points.append(np.array(points))
                        else:
                            find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, next_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)

    else:

        count = 0
        if len(next_idxs) > 2:
            for ii in range(0,len(next_idxs)):
                if next_idxs[ii,1] == 0:
                    next_vertex = subscripts[next_idxs[ii,0],1]-1
                else:
                    next_vertex = subscripts[next_idxs[ii,0],0]-1
                if (artery and art[next_vertex]) or ((not artery) and ven[next_vertex]):
                    count = count + 1

        if count < 3:

            for ii in range(0,len(next_idxs)):

                if next_idxs[ii,1] == 0:
                    next_vertex = subscripts[next_idxs[ii,0],1]-1
                else:
                    next_vertex = subscripts[next_idxs[ii,0],0]-1

                if next_vertex not in visited_points:

                    visited_points.append(next_vertex)

                    if artery == 2:
                        if art[next_vertex] != 0 or ven[next_vertex] != 0:
                            artery = art[next_vertex]

                        next_point = (vertices[next_vertex,0], vertices[next_vertex,1])
                        segment = LineString([previous_point, next_point])
                        points = segment.intersection(bbox_segments)
                        if points.geom_type == 'Point':
                            intersect_points.append(np.array(points))
                        else:
                            find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, next_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)

                    else:
                        if (artery and art[next_vertex]) or ((not artery) and ven[next_vertex]):
                            next_point = (vertices[next_vertex,0], vertices[next_vertex,1])
                            segment = LineString([previous_point, next_point])
                            points = segment.intersection(bbox_segments)
                            if points.geom_type == 'Point':
                                intersect_points.append(np.array(points))
                            else:
                                find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, next_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)


def intersect_connected(train,idx,bbox,selected_vertex):

    if train:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %idx)
    else:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
    bbox_segments = LineString([bbox[0],bbox[1],bbox[2],bbox[3],bbox[0]])
    intersect_points = []
    visited_points = []
    visited_points.append(selected_vertex)

    find_next_point(selected_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)
    return intersect_points

def intersect_connected_same_vessel(train,idx,bbox,selected_vertex, artery, bifurcations_allowed):

    if train:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %idx)
    else:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
    subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
    art = np.squeeze(mat_contents['G']['art'][0,0])
    ven = np.squeeze(mat_contents['G']['ven'][0,0])
    bbox_segments = LineString([bbox[0],bbox[1],bbox[2],bbox[3],bbox[0]])
    intersect_points = []
    visited_points = []
    visited_points.append(selected_vertex)

    find_next_point_same_vessel(bifurcations_allowed, artery, art, ven, selected_vertex, vertices, subscripts, bbox_segments, visited_points, intersect_points)
    return intersect_points


def find_junctions(train,idx,selected_vertex):

    if train:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/training/%02d_manual1.mat' %idx)
    else:
        mat_contents = sio.loadmat('./gt_dbs/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
    vertices = np.squeeze(mat_contents['G']['V'][0,0])-1 #positions seems to be (1,1) displaced
    junctions = np.squeeze(mat_contents['G']['junctions'][0,0])-1 #indexes in Python start in 0 instead of 1

    junction_points = []

    center_point = (vertices[selected_vertex,0], vertices[selected_vertex,1])
    min_x = center_point[0]-26
    max_x = center_point[0]+26
    min_y = center_point[1]-26
    max_y = center_point[1]+26

    for ii in range(0,len(junctions)):
        junction_point = (vertices[junctions[ii],0], vertices[junctions[ii],1])
        if junction_point[0] > min_x and junction_point[0] < max_x and junction_point[1] > min_y and junction_point[1] < max_y:
            junction_points.append(junction_point)


    return junction_points





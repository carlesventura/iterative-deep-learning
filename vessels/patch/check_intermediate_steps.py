__author__ = 'carlesv'

import bifurcations_toolbox as tb
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from shapely.geometry import LineString

img_id = 11
patch_id = 1

junctions = True

#Check img_crop
db_root_dir = '/scratch_net/boxy/carlesv/gt_dbs/DRIVE/'
num_patches_per_image = 50
num_images = 20
f = open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/vertices_selected_margin_6.txt','r')
found = False
for jj in range(0,num_patches_per_image):
    for ii in range(0,num_images):
        line = f.readline()
        if jj+1 == patch_id and ii+1 == img_id:
            found = True
            break

    if found:
        break

f.close()
selected_vertex = int(line.split()[1])
patch_size = 64
input_on_left_rotated = False

if junctions:
    img_crop, output_points = tb.find_junctions_selected_vertex(db_root_dir, selected_vertex, False, ii+1, patch_size)
else:
    img_crop, output_points = tb.find_output_points_selected_vertex(db_root_dir, selected_vertex, False, ii+1, patch_size, input_on_left_rotated)

if len(output_points)>0:
    output_points = np.vstack({tuple(row) for row in output_points}) #remove duplicated values on output_points
    output_points = np.round(output_points) #round intersect points to generate gaussians all centered on pixels

img_crop = np.array(img_crop, dtype=np.uint8)
plt.imshow(img_crop)
#plt.show(block=False)

#plt.figure()

#Check intersection points with borders
mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %img_id)
vertices = np.squeeze(mat_contents['G']['V'][0,0])-1
subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
center = (vertices[selected_vertex,0], vertices[selected_vertex,1])
x_tmp = int(center[0]-patch_size/2)
y_tmp = int(center[1]-patch_size/2)

for ii in range(0,len(subscripts)):
    segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
    xcoords, ycoords = segment.xy
    plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='blue', linewidth=1)
plt.xlim([0, patch_size-1])
plt.ylim([patch_size-1,0])
ax = plt.gca()
ax.set_aspect(1)

if not junctions:
    margin = int(np.round(patch_size/10.0))
    bbox = np.array([[center[0]-patch_size/2+margin,center[0]+patch_size/2-margin-1,center[0]+patch_size/2-margin-1,center[0]-patch_size/2+margin],
                     [center[1]-patch_size/2+margin,center[1]-patch_size/2+margin,center[1]+patch_size/2-margin-1,center[1]+patch_size/2-margin-1]])
    bbox = np.transpose(bbox)

    for ii in range(0,4):
        segment = LineString([bbox[ii%4], bbox[(ii+1)%4]])
        xcoords, ycoords = segment.xy
        plt.plot(xcoords-np.asarray(x_tmp), ycoords-np.asarray(y_tmp), color='green', linewidth=1)

for ii in range(0, len(output_points)):
    plt.plot(output_points[ii][0], output_points[ii][1], ls='none', color='green',marker='o', ms=10, lw=1.5, mfc='none')

plt.show(block=False)

plt.figure()

#Check generated ground truth from
gt = tb.make_gt(img_crop, output_points, (patch_size,patch_size), 1)
gt = np.squeeze(gt)
plt.imshow(gt)
plt.show()

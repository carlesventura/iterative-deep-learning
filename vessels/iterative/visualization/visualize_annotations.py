__author__ = 'carlesv'
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from skimage.morphology import skeletonize

idx = 1

pred_vessels = Image.open('/scratch_net/boxy/carlesv/HourGlasses_experiments/Iterative_margin_6/results_DRIU_vessel_segmentation/%02d_test.png' %(idx))
pred_vessels = np.array(pred_vessels)
pred_vessels = skeletonize(pred_vessels>100)
mat_contents = sio.loadmat('/scratch_net/boxy/carlesv/artery-vein/AV-DRIVE/test/%02d_manual1.mat' %idx)
vertices = np.squeeze(mat_contents['G']['V'][0,0])
subscripts = np.squeeze(mat_contents['G']['subscripts'][0,0])
plt.imshow(pred_vessels)
for ii in range(0,len(subscripts)):
    segment = LineString([vertices[subscripts[ii,0]-1]-1, vertices[subscripts[ii,1]-1]-1])
    xcoords, ycoords = segment.xy
    plt.plot(xcoords, ycoords, color='red', linewidth=1, solid_capstyle='round', zorder=2)
    # segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
    # xcoords, ycoords = segment.xy
    # plt.plot(xcoords, ycoords, color='blue', linewidth=1, solid_capstyle='round', zorder=2)
plt.show()

# img = Image.open('/scratch_net/boxy/carlesv/gt_dbs/DRIVE/test/images/%02d_test.tif' %(idx))
# plt.imshow(img)
# for ii in range(0,len(subscripts)):
#     segment = LineString([vertices[subscripts[ii,0]-1]-1, vertices[subscripts[ii,1]-1]-1])
#     xcoords, ycoords = segment.xy
#     plt.plot(xcoords, ycoords, color='green', linewidth=1, solid_capstyle='round', zorder=2)
#     segment = LineString([vertices[subscripts[ii,0]-1], vertices[subscripts[ii,1]-1]])
#     xcoords, ycoords = segment.xy
#     plt.plot(xcoords, ycoords, color='blue', linewidth=1, solid_capstyle='round', zorder=2)
# plt.show()
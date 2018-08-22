from PIL import Image
import numpy as np
import scipy.ndimage.morphology as ndimage
import matplotlib.pyplot as plt

img_id = 17

artery_img = Image.open('/scratch_net/boxy/carlesv/results/artery/iterative/%02d_test_dilated.png' % img_id)
vein_img = Image.open('/scratch_net/boxy/carlesv/results/vein/iterative/%02d_test_dilated.png' % img_id)
artery_img = np.array(artery_img)
vein_img = np.array(vein_img)

artery_dilated = ndimage.grey_dilation(artery_img, size=(3,3))
vein_dilated = ndimage.grey_dilation(vein_img, size=(3,3))

composed_img = np.zeros((584,565,3),dtype=np.uint8)
composed_img[:,:,0] = artery_dilated
composed_img[:,:,2] = vein_dilated

no_detections = (artery_dilated==0) * (vein_dilated==0)
indxs_no_detections = np.argwhere(no_detections==True)

composed_img[indxs_no_detections[:,0],indxs_no_detections[:,1],:] = 220
plt.imshow(composed_img)
plt.show()

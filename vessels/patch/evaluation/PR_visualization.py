__author__ = 'carlesv'
import numpy as np
import matplotlib.pyplot as plt

connected = True
from_same_vessel = False
bifurcations_allowed = True

if not connected:
    output_file = './results_dir_vessels/PR_not_connected.npz'
else:
    if from_same_vessel:
        if bifurcations_allowed:
            output_file = './results_dir_vessels/PR_connected_same_vessel.npz'
        else:
            output_file = './results_dir_vessels/PR_connected_same_vessel_wo_bifurcations.npz'
    else:
        output_file = './results_dir_vessels/PR_results_connected.npz'

curves = np.load(output_file)

recall_overall = curves['recall_overall']
precision_overall = curves['precision_overall']
recall_F_max = curves['recall_F_max']
precision_F_max = curves['precision_F_max']

plt.figure()
plt.plot(recall_overall,precision_overall, color='red', label='SHG [F=%02f]'%(2*recall_F_max*precision_F_max/(recall_F_max+precision_F_max)))
plt.plot(recall_F_max,precision_F_max,color='red',marker='+', ms=10)
print([precision_F_max, recall_F_max])

if not connected:
    output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation.npz'
else:
    if from_same_vessel:
        if bifurcations_allowed:
            output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_connected_same_vessel.npz'
        else:
            output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_connected_same_vessel_wo_bifurcations.npz'
    else:
        output_file = './results_dir_vessels/PR_DRIU_vessel_segmentation_results_connected.npz'

curves = np.load(output_file)

recall_overall = curves['recall_overall']
precision_overall = curves['precision_overall']
recall_F_max = curves['recall_F_max']
precision_F_max = curves['precision_F_max']

plt.plot(recall_overall,precision_overall, color='blue', label='DRIU [F=%02f]'%(2*recall_F_max*precision_F_max/(recall_F_max+precision_F_max)))
plt.plot(recall_F_max,precision_F_max,color='blue',marker='+', ms=10)
print([precision_F_max, recall_F_max])


plt.ylim([0,1])
plt.xlim([0,1])
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

__author__ = 'carlesv'
import numpy as np
import matplotlib.pyplot as plt

connected = True
from_same_vessel = False
bifurcations_allowed = True

output_file_not_connected = './results_dir_vessels/PR_not_connected.npz'
output_file_connected_same_vessel = './results_dir_vessels/PR_connected_same_vessel.npz'
output_file_connected = './results_dir_vessels/PR_results_connected.npz'

curves = np.load(output_file_not_connected)

recall_overall = curves['recall_overall']
precision_overall = curves['precision_overall']
recall_F_max = curves['recall_F_max']
precision_F_max = curves['precision_F_max']

plt.figure()
plt.plot(recall_overall,precision_overall, color='red', label='Not connected [F=%02f]'%(2*recall_F_max*precision_F_max/(recall_F_max+precision_F_max)))
plt.plot(recall_F_max,precision_F_max,color='red',marker='+', ms=10)
print([precision_F_max, recall_F_max])

curves = np.load(output_file_connected)

recall_overall = curves['recall_overall']
precision_overall = curves['precision_overall']
recall_F_max = curves['recall_F_max']
precision_F_max = curves['precision_F_max']

plt.plot(recall_overall,precision_overall, color='blue', label='Connected [F=%02f]'%(2*recall_F_max*precision_F_max/(recall_F_max+precision_F_max)))
plt.plot(recall_F_max,precision_F_max,color='blue',marker='+', ms=10)
print([precision_F_max, recall_F_max])

curves = np.load(output_file_connected_same_vessel)

recall_overall = curves['recall_overall']
precision_overall = curves['precision_overall']
recall_F_max = curves['recall_F_max']
precision_F_max = curves['precision_F_max']

plt.plot(recall_overall,precision_overall, color='green', label='Connected same vessel [F=%02f]'%(2*recall_F_max*precision_F_max/(recall_F_max+precision_F_max)))
plt.plot(recall_F_max,precision_F_max,color='green',marker='+', ms=10)
print([precision_F_max, recall_F_max])

plt.ylim([0,1])
plt.xlim([0,1])
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.show()

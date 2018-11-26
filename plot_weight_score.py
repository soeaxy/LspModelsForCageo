import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
###############################################################################
# Load a dataset
###############################################################################
data = pd.read_csv('data\weight_file.csv')
name_list = ['AUC','Mean_Recall']
weight = data['weight']
AUC = data['balanced_accuracy_score']
Recall = data['geometric_mean_score']

max_AUC=np.argmax(AUC)
max_Recall=np.argmax(Recall)

plt.plot(weight,AUC,'r-*',label=name_list[0])
plt.plot(max_AUC+1,AUC[max_AUC])

plt.plot(weight,Recall,'g-.',label=name_list[1])
plt.plot(max_Recall+1,Recall[max_Recall])
show_max='Best Weight: '+str(max_Recall+1)

plt.text(20, 0.875, show_max, fontsize=12)
plt.legend()
plt.axvline(x=max_AUC+1, color='b', linewidth=1)
plt.xlabel('Weight of the landslide samples data')
plt.ylabel('Score of AUC and Mean_Recall')
plt.show()
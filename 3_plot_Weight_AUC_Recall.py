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

# Find max value
max_AUC=np.argmax(AUC)
max_Recall=np.argmax(Recall)

plt.plot(weight,AUC,'r-*',label=name_list[0])
plt.plot(max_AUC+1,AUC[max_AUC],'gs')

plt.plot(weight,Recall,'g-o',label=name_list[1])
plt.plot(max_Recall+1,Recall[max_Recall],'bs')
show_max='Best Weight: '+str(max_Recall+1) + '\n'+f'AUC: {AUC[max_AUC]}' + '\n' + f'Recall: {Recall[max_Recall]} '

plt.annotate(
        show_max, 
        xy = (max_Recall+1,Recall[max_Recall]), 
        xycoords='data',
        xytext = (25,0.8),
        textcoords = 'data', ha = 'center', va = 'center',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.axvline(x=max_AUC+1, color='b', linestyle=':', linewidth=1, label='Best weight chosen')
plt.xlabel('Weight of the landslide samples data')
plt.ylabel('Score of AUC and Mean_Recall')
plt.legend()
plt.show()
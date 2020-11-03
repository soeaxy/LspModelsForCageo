import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
###############################################################################
# Load a dataset
###############################################################################
data = pd.read_csv('data\weight_file.csv')
name_list = ['Balanced accuracy','geometric_mean_score','Recall','AUC']
weight = data['weight']
Balanced_acc_score = data['balanced_accuracy_score']
Geo_score = data['geometric_mean_score']
Recall = data['recall_score']
AUC = data['AUC']

# Find max value
max_Geo=np.argmax(Geo_score)
max_Recall=np.argmax(Recall)

plt.plot(weight,Geo_score,'r-*',label='Geometric Mean Score')
plt.plot(weight,Recall,'g-o',label=name_list[2])
plt.plot(weight,AUC,'b-*',label=name_list[3])

plt.plot(max_Geo+1,Geo_score[max_Geo],'gs')

# plt.plot(max_Recall+1,Recall[max_Recall],'bs')
show_max='Best Weight: '+str(max_Geo+1) + '\n'+f'Geometric Mean Score: {round(Geo_score[max_Geo],3)}'

plt.annotate(
        show_max, 
        xy = (max_Geo+1,Geo_score[max_Geo]), 
        xycoords='data',
        xytext = (24,0.65),
        textcoords = 'data', ha = 'center', va = 'center',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.axvline(x=max_Geo+1, color='b', linestyle=':', linewidth=1, label='Best weight chosen')
plt.xlabel('Weight of the landslide samples data')
plt.ylabel('Score')
plt.legend()
plt.show()
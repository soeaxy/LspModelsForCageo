'''
Author: yxsong
Date: 2021-09-10 09:28:16
LastEditTime: 2021-09-16 11:09:26
LastEditors: yxsong
Description: 
FilePath: \LSM_Model_Imbalanced\WeightCalculate.py
 
'''
'''
ref: http://www.itkeyword.com/doc/7731032350828881x509/sklearn-pipeline-applying-sample-weights-after-applying-a-polynomial-feature-t
'''
import itertools
from os import name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# DataPrepare
def DataPrepare(data):
    target = 'value'
    IDCol = 'ID'
    GeoID = data[IDCol]
    print(data[target].value_counts())
    colName = ['Elevation', 'Slope', 'Aspect', 'TRI', 'Curvature', 'Lithology', 'River', 'NDVI', 'NDWI', 'Rainfall', 'Earthquake', 'Land_use']
    X = data[colName]
    y = data[target]
    return X, y, GeoID

# Change weight for the sample and save the weight
def WriteWeihtFile(X,y, classifier, weight_file):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4)
    # Instanciate a PCA object
    pca = PCA(n_components='mle')
    stdscaler = preprocessing.StandardScaler()
    with open(weight_file, 'w') as f:
            f.write('%s,%s,%s,%s,%s\n'%('weight','balanced_accuracy_score','geometric_mean_score','recall_score','AUC'))
            for i in range(1,31):
                    class_weight = {0:1, 1:i}
                    blgb = classifier(random_state=0, n_jobs=-1, class_weight=class_weight)
                    pipeline_blgb = Pipeline([('STDSCALE', stdscaler), ('PCA', pca), ('BLGB', blgb)])
                    pipeline_blgb.fit(X_train, y_train)
                    y_pred_blgb = pipeline_blgb.predict(X_test)
                    y_pred_blgb_p = pipeline_blgb.predict_proba(X_test)[:, 1]
                    print(f'Weight is {i}: '+'Weighted' + str(classifier) +'classifier performance:')
                    print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - AUC {:.3f}'
                .format(balanced_accuracy_score(y_test, y_pred_blgb),
                        geometric_mean_score(y_test, y_pred_blgb), recall_score(y_test, y_pred_blgb), roc_auc_score(y_test,y_pred_blgb_p)))
                
                    f.write(f'{i}, {balanced_accuracy_score(y_test, y_pred_blgb)}, {geometric_mean_score(y_test, y_pred_blgb)}, {recall_score(y_test, y_pred_blgb)}, {roc_auc_score(y_test,y_pred_blgb_p)}'+'\n')
    print('Done')
    f.close


def PlotWeightAndRecall(inputFile):
    data = pd.read_csv(inputFile)
    name_list = ['Balanced accuracy','geometric_mean_score','Recall','AUC']
    weight = data['weight']
    Balanced_acc_score = data['balanced_accuracy_score']
    Geo_score = data['geometric_mean_score']
    Recall = data['recall_score']
    AUC = data['AUC']

    max_Geo=np.argmax(Geo_score)
    max_Recall=np.argmax(Recall)

    plt.plot(weight,Geo_score,'r-*',label='Geometric Mean Score')
    # plt.plot(weight,Recall,'g-o',label=name_list[2])
    # plt.plot(weight,AUC,'b-*',label=name_list[3])

    plt.plot(max_Geo+1,Geo_score[max_Geo],'gs')

    show_max='Best Weight: '+str(max_Geo+1) + '\n'+f'Geometric Mean Score: {round(Geo_score[max_Geo],3)}'

    plt.annotate(
        show_max, 
        xy = (max_Geo+1,Geo_score[max_Geo]), 
        xycoords='data',
        xytext = (24,0.65),
        textcoords = 'data', ha = 'center', va = 'center',
        bbox = dict(boxstyle = 'round, pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=0'))

    plt.axvline(x=max_Geo+1, color='b', linestyle=':', linewidth=1, label='Best weight chosen')
    plt.xlabel('Weight of the landslide samples data')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # data = pd.read_csv('./data/wanzhou_island.csv')
    # X, y, GeoID = DataPrepare(data)
    weight_file = r'data\weight_file.csv'
    # WriteWeihtFile(X, y, LGBMClassifier, weight_file)
    PlotWeightAndRecall(weight_file)

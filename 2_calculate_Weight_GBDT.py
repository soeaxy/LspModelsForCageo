'''
ref: http://www.itkeyword.com/doc/7731032350828881x509/sklearn-pipeline-applying-sample-weights-after-applying-a-polynomial-feature-t
'''
import itertools

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

###############################################################################
# Load an imbalanced dataset
###############################################################################

# Data prepare
def data_raw(data):
    target = 'value'
    IDCol = 'ID'
    GeoID = data[IDCol]
    print(data[target].value_counts())
    x_columns = [x for x in data.columns if x not in [target,IDCol,'GRID_CODE']]
    X = data[x_columns]
    y = data[target]
    return X, y, GeoID

data = pd.read_csv('./data/wanzhou_island.csv')
X, y, GeoID = data_raw(data)
X = preprocessing.scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Instanciate a PCA object
pca = PCA(n_components='mle')

# Instanciate a StandardScaler object
stdscaler = preprocessing.StandardScaler()

###############################################################################
# Classification using lightGBM classifier with different sample weight
###############################################################################

blgb = LGBMClassifier(random_state=0, n_jobs=-1)

# Add one transformers and a sampler in the pipeline object
pipeline_blgb = Pipeline([('STDSCALE', stdscaler), ('PCA', pca), ('BLGB', blgb)])

# Change weight for the sample and save the weight
weight_file = r'data\weight_file.csv'
with open(weight_file, 'w') as f:
        f.write('%s,%s,%s,%s\n'%('weight','balanced_accuracy_score','geometric_mean_score','recall_score'))
        for i in range(1,31):
                sample_weight = [i if y == 1 else 1 for y in y_train]
                pipeline_blgb.fit(X_train, y_train, **{'BLGB__sample_weight': sample_weight})
                y_pred_blgb = pipeline_blgb.predict(X_test)

                print(f'Weight is {i}: '+'Weighted GBDT classifier performance:')
                print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - AUC {:.3f}'
                .format(balanced_accuracy_score(y_test, y_pred_blgb),
                        geometric_mean_score(y_test, y_pred_blgb), recall_score(y_test, y_pred_blgb), roc_auc_score(y_test,y_pred_blgb)))
                
                f.write(f'{i}, {balanced_accuracy_score(y_test, y_pred_blgb)}, {geometric_mean_score(y_test, y_pred_blgb)},{recall_score(y_test, y_pred_blgb)}'+'\n')
print('Done')
f.close

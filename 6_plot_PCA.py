import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition as sk_decomposition
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


# Data prepare
def data_raw(data):
    target = 'value'
    IDCol = 'ID'
    GeoID = data[IDCol]
    print(data[target].value_counts())
    colName = ['Elevation', 'Slope', 'Aspect', 'TRI', 'Curvature', 'Lithology', 'River', 'NDVI', 'NDWI', 'Rainfall', 'Earthquake', 'Land_use']
    X = data[colName]
    y = data[target]
    return X, y, GeoID

data = pd.read_csv('./data/wanzhou_island.csv')
X, y, GeoID = data_raw(data)

pca = sk_decomposition.PCA(n_components='mle',whiten=True,svd_solver='auto')
pca.fit(X)
reduced_X = pca.transform(X) #reduced_X为降维后的数据
print('PCA:')
print ('降维后的各主成分的方差值占总方差值的比例',pca.explained_variance_ratio_)
print ('降维后的各主成分的方差值',pca.explained_variance_)
print ('降维后的特征数',pca.n_components_)

# 作者：BlackBlog__
# 链接：https://www.jianshu.com/p/731610dca805
# 來源：简书
# 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。

def Normalize(data):
    m = np.mean(data)
    mx = max(data)
    mn = min(data)
    s = sum(data)
    return [float(i) / s for i in data]

f_importance = [0.082,0.074,0.073,0.065,0.062,0.046,0.041,0.040]

x = Normalize(f_importance)
print(x)
s = ['X27','X28','X17','X1','X24','X6','X21','X3']

plt.xlabel('Selected features')
plt.ylabel('Feature importances')
plt.bar(s,x)
plt.show()
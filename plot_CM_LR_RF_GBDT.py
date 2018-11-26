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


def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

def data_raw(data):
    target = 'value'
    IDCol = 'ID'
    GeoID = data[IDCol]
    print(data[target].value_counts())
    x_columns = [x for x in data.columns if x not in [target,IDCol,'GRID_CODE']]
    X = data[x_columns]
    y = data[target]
    return X, y, GeoID
###############################################################################
# Load an imbalanced dataset
###############################################################################

data = pd.read_csv('./data/wanzhou_island.csv')
X, y, GeoID = data_raw(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Instanciate a PCA object
pca = PCA(n_components='mle')

###############################################################################
# Classification using LogisticRegressionCV classifier with and without sampling
###############################################################################

lr = LogisticRegressionCV(cv=5, random_state=0)
blr = LogisticRegressionCV(cv=5, random_state=0, class_weight='balanced')

# Add one transformers and a sampler in the pipeline object
pipeline_lr = make_pipeline(pca, lr)
pipeline_blr = make_pipeline(pca, blr)

pipeline_lr.fit(X_train, y_train)
pipeline_blr.fit(X_train, y_train)

y_pred_lr = pipeline_lr.predict(X_test)
y_pred_blr = pipeline_blr.predict(X_test)

print('Logistic Regression classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test, y_pred_lr),
              geometric_mean_score(y_test, y_pred_lr)))
cm_rf = confusion_matrix(y_test, y_pred_lr)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rf, classes=[0,1], ax=ax[0],
                      title='Logistic Regression')

print('Balanced Logistic Regression classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test, y_pred_blr),
              geometric_mean_score(y_test, y_pred_blr)))
cm_brf = confusion_matrix(y_test, y_pred_blr)
plot_confusion_matrix(cm_brf, classes=[0,1], ax=ax[1],
                      title='Balanced Logistic Regression')

###############################################################################
# Classification using random forest classifier with and without sampling
###############################################################################
# Random forest is another popular ensemble method and it is usually
# outperforming bagging. Here, we used a vanilla random forest and its balanced
# counterpart in which each bootstrap sample is balanced.

rf = RandomForestClassifier(random_state=0, n_jobs=-1)
brf = BalancedRandomForestClassifier(random_state=0, n_jobs=-1)

# Add one transformers and a sampler in the pipeline object
pipeline_rf = make_pipeline(pca, rf)
pipeline_brf = make_pipeline(pca, brf)

pipeline_rf.fit(X_train, y_train)
pipeline_brf.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)
y_pred_brf = pipeline_brf.predict(X_test)

# Similarly to the previous experiment, the balanced classifier outperform the
# classifier which learn from imbalanced bootstrap samples. In addition, random
# forest outsperforms the bagging classifier.

print('Random Forest classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test, y_pred_rf),
              geometric_mean_score(y_test, y_pred_rf)))
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rf, classes=[0,1], ax=ax[0],
                      title='Random forest')

print('Balanced Random Forest classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test, y_pred_brf),
              geometric_mean_score(y_test, y_pred_brf)))
cm_brf = confusion_matrix(y_test, y_pred_brf)
plot_confusion_matrix(cm_brf, classes=[0,1], ax=ax[1],
                      title='Balanced random forest')

###############################################################################
# Classification using lightGBM classifier with and without sampling
###############################################################################

lgb = LGBMClassifier(random_state=0, n_jobs=-1)
blgb = LGBMClassifier(random_state=0, n_jobs=-1)

# Add one transformers and a sampler in the pipeline object
pipeline_lgb = make_pipeline(pca, lgb)
pipeline_blgb = make_pipeline(pca, blgb)

lgb.fit(X_train, y_train)
blgb.fit(X_train, y_train, sample_weight=[18 if y == 1 else 1 for y in y_train])

y_pred_lgb = lgb.predict(X_test)
y_pred_blgb = blgb.predict(X_test)

print('GBDT classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - AUC {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_lgb),
              geometric_mean_score(y_test, y_pred_lgb), recall_score(y_test, y_pred_lgb), roc_auc_score(y_test,y_pred_lgb)))

# print('GBDT\n',classification_report(y_test, y_pred_lgb, target_names=['No_Landslide','Landslide']))
cm_lgb = confusion_matrix(y_test, y_pred_lgb)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_lgb, classes=[0,1], ax=ax[0],
                      title='GBDT')

print('Weighted GBDT classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - AUC {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_blgb),
              geometric_mean_score(y_test, y_pred_blgb), recall_score(y_test, y_pred_blgb), roc_auc_score(y_test,y_pred_blgb)))
# print('weighted GBDT\n',classification_report(y_test, y_pred_blgb, target_names=['No_Landslide','Landslide']))

cm_blgb = confusion_matrix(y_test, y_pred_blgb)
plot_confusion_matrix(cm_blgb, classes=[0,1], ax=ax[1],
                      title='Weighted GBDT')

plt.show()

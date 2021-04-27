import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.datasets import fetch_datasets
from imblearn.ensemble import (BalancedBaggingClassifier,
                               BalancedRandomForestClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)
from imblearn.metrics import geometric_mean_score
from imblearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from sklearn import datasets, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

################
#Data Prepare
################
train = pd.read_csv('data/wanzhou_island.csv')
train_labels = train['value']
x_columns = ['Elevation', 'Slope', 'Aspect', 'TRI', 'Curvature', 'Lithology', 'River', 'NDVI', 'NDWI', 'Rainfall', 'Earthquake', 'Land_use']
GeoID = train['ID']

X, y = train[x_columns], train_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Instanciate a PCA object
pca = PCA(n_components='mle')

# Instanciate a StandardScaler object
stdscaler = preprocessing.StandardScaler()

# Classification using LogisticRegressionCV classifier with and without sampling

lr = LogisticRegressionCV(cv=5, random_state=0)
wlr = LogisticRegressionCV(cv=5, random_state=0, class_weight={0:1, 1:19})

# Add one transformers and a sampler in the pipeline object
pipeline_lr = make_pipeline(stdscaler, pca, lr)
pipeline_wlr = make_pipeline(stdscaler, pca, wlr)

pipeline_lr.fit(X_train, y_train)
pipeline_wlr.fit(X_train, y_train)

y_pred_lr = pipeline_lr.predict(X_test)
y_pred_wlr = pipeline_wlr.predict(X_test)

print('Logistic Regression classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_lr),
              geometric_mean_score(y_test, y_pred_lr),recall_score(y_test, y_pred_lr),accuracy_score(y_test, y_pred_lr)))
cm_rf = confusion_matrix(y_test, y_pred_lr)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rf, classes=[0,1], ax=ax[0],
                      title='Logistic Regression')

print('Balanced Logistic Regression classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f}- Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_wlr),
              geometric_mean_score(y_test, y_pred_wlr),recall_score(y_test, y_pred_wlr),accuracy_score(y_test, y_pred_wlr)))
cm_wrf = confusion_matrix(y_test, y_pred_wlr)
plot_confusion_matrix(cm_wrf, classes=[0,1], ax=ax[1],
                      title='Weighted Logistic Regression')
plt.show()

###############################################################################
# Classification using random forest classifier with and without sampling
###############################################################################

rf = RandomForestClassifier(random_state=0)
wrf = BalancedRandomForestClassifier(random_state=0)

# Add one transformers and a sampler in the pipeline object
pipeline_rf = make_pipeline(stdscaler, pca, rf)
pipeline_wrf = make_pipeline(stdscaler, pca, wrf)

pipeline_rf.fit(X_train, y_train)
pipeline_wrf.fit(X_train, y_train)

y_pred_rf = pipeline_rf.predict(X_test)
y_pred_wrf = pipeline_wrf.predict(X_test)

# Similarly to the previous experiment, the balanced classifier outperform the
# classifier which learn from imbalanced bootstrap samples. In addition, random
# forest outsperforms the bagging classifier.

print('Random Forest classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_rf),
              geometric_mean_score(y_test, y_pred_rf), recall_score(y_test, y_pred_rf),accuracy_score(y_test, y_pred_rf)))
cm_rf = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rf, classes=[0,1], ax=ax[0],
                      title='RF')

print('Balanced Random Forest classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f}- Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_wrf),
              geometric_mean_score(y_test, y_pred_wrf),recall_score(y_test, y_pred_wrf),accuracy_score(y_test, y_pred_wrf)))
cm_wrf = confusion_matrix(y_test, y_pred_wrf)
plot_confusion_matrix(cm_wrf, classes=[0,1], ax=ax[1], title='WRF')

plt.show()
###############################################################################
# Classification using lightGBM classifier with and without sampling
###############################################################################

lgb = LGBMClassifier(random_state=0, n_jobs=-1)
wlgb = LGBMClassifier(random_state=0, n_jobs=-1, class_weight={0:1, 1:19})

# Add one transformers and a sampler in the pipeline object
pipeline_lgb = make_pipeline(stdscaler, pca, lgb)
pipeline_wlgb = make_pipeline(stdscaler, pca, wlgb)

pipeline_lgb.fit(X_train, y_train)
pipeline_wlgb.fit(X_train, y_train)

y_pred_lgb = pipeline_lgb.predict(X_test)
y_pred_wlgb = pipeline_wlgb.predict(X_test)

print('LightGBM classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_lgb),
              geometric_mean_score(y_test, y_pred_lgb), recall_score(y_test, y_pred_lgb), accuracy_score(y_test,y_pred_lgb)))

cm_lgb = confusion_matrix(y_test, y_pred_lgb)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_lgb, classes=[0,1], ax=ax[0],
                      title='LightGBM')

print('Weighted LightGBM classifier performance:')
print('Balanced accuracy: {:.3f} - Geometric mean {:.3f} - Recall {:.3f} - Accuracy {:.3f}'
      .format(balanced_accuracy_score(y_test, y_pred_wlgb),
              geometric_mean_score(y_test, y_pred_wlgb), recall_score(y_test, y_pred_wlgb), accuracy_score(y_test,y_pred_wlgb)))

cm_wlgb = confusion_matrix(y_test, y_pred_wlgb)
plot_confusion_matrix(cm_wlgb, classes=[0,1], ax=ax[1],
                      title='WLightGBM')
plt.show()
###############################################################################
# Plot ROC Curve
###############################################################################

# The lr model 
y_pred_lr_p = pipeline_lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr_p)
auc_lr = roc_auc_score(y_test,y_pred_lr_p)

# The weighted lr model 
y_pred_wlr_p = pipeline_wlr.predict_proba(X_test)[:, 1]
fpr_wlr, tpr_wlr, _ = roc_curve(y_test, y_pred_wlr_p)
auc_wlr = roc_auc_score(y_test,y_pred_wlr_p)

# The lightgmb model 
y_pred_lgb_p = pipeline_lgb.predict_proba(X_test)[:, 1]
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_lgb_p)
auc_lgb = roc_auc_score(y_test,y_pred_lgb_p)

# The weighted lightgmb model 
y_pred_wlgb_p = pipeline_wlgb.predict_proba(X_test)[:, 1]
fpr_wlgb, tpr_wlgb, _ = roc_curve(y_test, y_pred_wlgb_p)
auc_wlgb = roc_auc_score(y_test,y_pred_wlgb_p)

# The rf model 
y_pred_rf_p = pipeline_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf_p)
auc_rf = roc_auc_score(y_test,y_pred_rf_p)

# The weighted rf model 
y_pred_wrf_p = pipeline_wrf.predict_proba(X_test)[:, 1]
fpr_wrf, tpr_wrf, _ = roc_curve(y_test, y_pred_wrf_p)
auc_wrf = roc_auc_score(y_test,y_pred_wrf_p)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR (AUC=%0.3f)' % (auc_lr), lw=2)
plt.plot(fpr_wlr, tpr_wlr, label='WLR (AUC=%0.3f)' % (auc_wlr), lw=2)

plt.plot(fpr_lgb, tpr_lgb, label='LightGBM (AUC=%0.3f)' % (auc_lgb), lw=2)
plt.plot(fpr_wlgb, tpr_wlgb, label='WLightGBM(AUC=%0.3f)' % (auc_wlgb), lw=2)

plt.plot(fpr_rf, tpr_rf, label='RF (AUC=%0.3f)' % (auc_rf), lw=2)
plt.plot(fpr_wrf, tpr_wrf, label='WRF(AUC=%0.3f)' % (auc_wrf), lw=2)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

plt.show()

###############################################################################
# Save Models and Results
###############################################################################

# save results
def save_results(GeoID, y_pred, y_predprob, result_file):
    results = np.vstack((GeoID,y_pred,y_predprob))
    results = np.transpose(results)
    header_string = 'GeoID, y_pred, y_predprob'
    np.savetxt(result_file, results, header = header_string, fmt = '%d,%d,%0.5f',delimiter = ',')
    print('Saving file Done!')

y_pred_gdbt = pipeline_lgb.predict(X)
y_pred_proba_gdbt = pipeline_lgb.predict_proba(X)[:,1]
result_file_gdbt = './data/lightgbm.txt'
save_results(GeoID, y_pred_gdbt, y_pred_proba_gdbt, result_file_gdbt)

y_pred_gdbt_weighted = pipeline_wlgb.predict(X)
y_pred_proba_gdbt_weighted = pipeline_wlgb.predict_proba(X)[:,1]
result_file_gdbt_weighted = './data/wlightgbm.txt'
save_results(GeoID, y_pred_gdbt_weighted, y_pred_proba_gdbt_weighted, result_file_gdbt_weighted)

y_pred_lr = pipeline_lr.predict(X)
y_pred_proba_lr = pipeline_lr.predict_proba(X)[:,1]
result_file_lr = './data/lr.txt'
save_results(GeoID, y_pred_lr, y_pred_proba_lr, result_file_lr)

y_pred_wlr = pipeline_wlr.predict(X)
y_pred_proba_wlr = pipeline_wlr.predict_proba(X)[:,1]
result_file_wlr = './data/wlr.txt'
save_results(GeoID, y_pred_wlr, y_pred_proba_wlr, result_file_wlr)

y_pred_rf = pipeline_rf.predict(X)
y_pred_proba_rf = pipeline_rf.predict_proba(X)[:,1]
result_file_rf = './data/rf.txt'
save_results(GeoID, y_pred_rf, y_pred_proba_rf, result_file_rf)

y_pred_wrf = pipeline_wrf.predict(X)
y_pred_proba_wrf = pipeline_wrf.predict_proba(X)[:,1]
result_file_wrf = './data/wrf.txt'
save_results(GeoID, y_pred_wrf, y_pred_proba_wrf, result_file_wrf)

print('Done!')
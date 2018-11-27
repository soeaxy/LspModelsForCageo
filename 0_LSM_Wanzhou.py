# -*- coding: utf-8 -*-
"""
Created on Tue 8 10 2018
@author : Yxsong
@email  : yxsong@cug.edu.cn

what does the doc do?
    some ideas of improving the accuracy of imbalanced data classification.
data characteristics:
    imbalanced data.
the models:
    model_baseline : lgb
    model_baseline3 : bagging
"""

import itertools
import pickle
import time
from collections import Counter
from random import shuffle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperopt import (STATUS_OK, Trials, fmin, hp, partial, rand, space_eval,
                      tpe)
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsemble
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy import interp
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, log_loss, recall_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier


def read_bigcsv(filename, **kw):
    with open(filename) as rf:
        reader = pd.read_csv(rf, **kw, iterator=True)
        chunkSize = 100000
        chunks = []
        while True:
            try:
                chunk = reader.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                print("Iteration is stopped.")
                break
        df = pd.concat(chunks, axis=0, join='outer', ignore_index=True)
    return df

def timestamp2datetime(value):
    value = time.localtime(value)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', value)
    return dt

def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def model_baseline(x_train, y_train, x_test, y_test):

    target_names = ['No_Landslide','Landslide']
    print("begin train.............................................................")

    # Logistic Regression
    kw_lr = dict(class_weight='balanced')
       
    lr = LogisticRegressionCV(**kw_lr)
    lr.fit(x_train, y_train)
    prob = lr.predict_proba(x_test,)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    
    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    print('Logistic Regression\n',classification_report(y_test, y_pred, target_names=target_names))
    accuracy_val = accuracy_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    
    fig = plt.figure('fig1')
    ax = fig.add_subplot(1, 1, 1)
    name = 'LR'
    plt.plot(mean_fpr, mean_tpr, 
             label='{} (AUC=%0.3f, Recall=%0.3f)'.format(name) %
             (x_auc, recall_val), lw=2)


    cm1 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1], title='')

    # 以写二进制的方式打开文件
    file = open("./model_lr.pickle", "wb")
    # 把模型写入到文件中
    pickle.dump(lr, file)
    # 关闭文件
    file.close()

    # lgb
    kw_lgb = {'learning_rate': 0.09, 'max_depth': 15, 'min_child_weight': 3, 'n_estimators': 70, 'subsample': 1}
    # clf = lgb.LGBMClassifier(**kw_lgb)
    clf = lgb.LGBMClassifier()

    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test,)[:, 1]
    predict_score = [float('%.3f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)

    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    print('GBDT\n'+classification_report(y_test, y_pred, target_names=target_names))
    accuracy_val = accuracy_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)

    
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    
    fig = plt.figure('fig1')
    ax = fig.add_subplot(1, 1, 1)
    name = 'GBDT'
    plt.plot(mean_fpr, mean_tpr, 
             label='{} (AUC=%0.3f, Recall=%0.3f)'.format(name) %
             (x_auc, recall_val), lw=2)
    y_pred = clf.predict(x_test)
    cm0 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1], title='')

    # 以写二进制的方式打开文件
    file = open("./model_GBDT.pickle", "wb")
    # 把模型写入到文件中
    pickle.dump(clf, file)
    # 关闭文件
    file.close()
    

    # add weighted according to the labels
    clf = lgb.LGBMClassifier(class_weight='balanced')

    clf.fit(x_train, y_train, sample_weight=[1 if y == 1 else 0.19 for y in y_train])

    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test,)[:, 1]
    predict_score = [float('%.4f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)

    y_pred = clf.predict(x_test)
    accuracy_val = accuracy_score(y_test, y_pred)
    recall_val = recall_score(y_test, y_pred)


    print('weighted GBDT\n'+classification_report(y_test, y_pred, target_names=target_names))

    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    name = 'weighted GBDT'
    plt.figure('fig1')  # 选择图
    plt.plot(
            mean_fpr, mean_tpr,
             label='{} (AUC=%0.3f, Recall=%0.3f)'.format(name) %
             (x_auc, recall_val), lw=2)

    cm2 = plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=[0, 1],
                          title='')
    plt.figure('fig1')  # 选择图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
    # make nice plotting
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.spines['left'].set_position(('outward', 10))
    # ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="best")
    plt.show()

    # 以写二进制的方式打开文件
    file = open("./model_weighted_GBDT.pickle", "wb")
    # 把模型写入到文件中
    pickle.dump(clf, file)
    # 关闭文件
    file.close()
    return cm0, cm1, cm2, fig
    
def model_baseline3(x_train, y_train, x_test, y_test):
    bagging = BaggingClassifier(random_state=0)
    balanced_bagging = BalancedBaggingClassifier(random_state=0)
    bagging.fit(x_train, y_train)
    balanced_bagging.fit(x_train, y_train)
    prob = bagging.predict_proba(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    y_pred = [1 if x > 0.5 else 0 for x in predict_score]
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    fig = plt.figure('Bagging')
    ax = fig.add_subplot(1, 1, 1)
    name = 'base_Bagging'
    plt.plot(mean_fpr, mean_tpr, linestyle='--',
             label='{} (AUC = %0.2f, logloss = %0.2f)'.format(name) %
             (x_auc, loss_val), lw=2)
    y_pred_bagging = bagging.predict(x_test)
    cm_bagging = confusion_matrix(y_test, y_pred_bagging)
    cm1 = plt.figure()
    plot_confusion_matrix(cm_bagging,
                          classes=[0, 1],
                          title='Confusion matrix of BaggingClassifier')
    # balanced_bagging
    prob = balanced_bagging.predict_proba(x_test)[:, 1]
    predict_score = [float('%.2f' % x) for x in prob]
    loss_val = log_loss(y_test, predict_score)
    fpr, tpr, thresholds = roc_curve(y_test, predict_score)
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = interp(mean_fpr, fpr, tpr)
    x_auc = auc(fpr, tpr)
    plt.figure('Bagging')  # 选择图
    name = 'base_Balanced_Bagging'
    plt.plot(
            mean_fpr, mean_tpr, linestyle='--',
            label='{} (AUC = %0.2f, logloss = %0.2f)'.format(name) %
            (x_auc, loss_val), lw=2)
    y_pred_balanced_bagging = balanced_bagging.predict(x_test)


    cm_balanced_bagging = confusion_matrix(y_test, y_pred_balanced_bagging)
    cm2 = plt.figure()
    plot_confusion_matrix(cm_balanced_bagging,
                          classes=[0, 1],
                          title='Confusion matrix of BalancedBagging')
    plt.figure('Bagging')  # 选择图
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck')
    
    # make nice plotting
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # ax.spines['left'].set_position(('outward', 10))
    # ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return cm1, cm2, fig


def data_raw():
    train = pd.read_csv('data\wanzhou_island.csv')
    target = 'value'
    IDCol = 'ID'
    GeoID = train[IDCol]
    print(train[target].value_counts())
    x_columns = [x for x in train.columns if x not in [target,IDCol,'GRID_CODE']]
    X = train[x_columns]
    y = train[target]
    return X, y, GeoID

def data_prepare(method = 'no_resample'):
    X, y ,GeoID = data_raw()

    X = preprocessing.scale(X)

    # EasyEnsemble 降采样
    ee = EasyEnsemble(random_state=50)
    X_ee, y_ee = ee.fit_sample(X, y)

    # RandomUnderSampler 降采样
    rus = RandomUnderSampler(random_state=50)
    X_rus, y_rus = rus.fit_sample(X, y)

    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(random_state=50)
    X_smo, y_smo = smo.fit_sample(X, y)

    if method == 'no_resample':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0,stratify=y)
        return X_train, X_test, y_train, y_test
    elif str(method) == 'smo':
        X_train, X_test, y_train, y_test = train_test_split(X_smo, y_smo, test_size=.3, random_state=0)
        return X_train, X_test, y_train, y_test
    elif str(method)=='rus':
        X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=.3, random_state=0)
        return X_train, X_test, y_train, y_test
    elif str(method)=='ee':
        X_train, X_test, y_train, y_test = train_test_split(X_ee, y_ee, test_size=.3, random_state=0)
        return X_train, X_test, y_train, y_test
    
# save results
def save_results(GeoID, y_pred, y_predprob, result_file):
    results = np.vstack((GeoID,y_pred,y_predprob))
    results = np.transpose(results)
    header_string = 'GeoID, y_pred, y_predprob'
    np.savetxt(result_file, results, header = header_string, fmt = '%d,%d,%0.5f',delimiter = ',')
    print('Saving file Done!')

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_prepare()
    cm10, cm11, cm12, fig1 = model_baseline(x_train, y_train, x_test, y_test)
    # cm31, cm32, fig3 = model_baseline3(x_train, y_train, x_test, y_test)

    cm10.savefig('./logistic_regression_ConMax.pdf', format='pdf')
    fig1.savefig('./base_lgb_weighted.pdf', format='pdf')
    cm11.savefig('./Confusion matrix1.pdf', format='pdf')
    cm12.savefig('./Confusion matrix2.pdf', format='pdf')

    # 以读二进制的方式打开文件
    file = open("./model_GBDT.pickle", "rb")
    # 把模型从文件中读取出来
    gdbt_model = pickle.load(file)
    # 关闭文件
    file.close()

    X, y, GeoID= data_raw()
    y_pred_gdbt = gdbt_model.predict(X)
    y_pred_proba_gdbt = gdbt_model.predict_proba(X)[:,1]
    result_file_gdbt = './gbdt.txt'
    save_results(GeoID, y_pred_gdbt, y_pred_proba_gdbt, result_file_gdbt)

    # 以读二进制的方式打开文件
    file = open("./model_weighted_GBDT.pickle", "rb")
    # 把模型从文件中读取出来
    gdbt_weighted_model = pickle.load(file)
    # 关闭文件
    file.close()

    y_pred_gdbt_weighted = gdbt_weighted_model.predict(X)
    y_pred_proba_gdbt_weighted = gdbt_weighted_model.predict_proba(X)[:,1]
    result_file_gdbt_weighted = './gdbt_weighted.txt'
    save_results(GeoID, y_pred_gdbt_weighted, y_pred_proba_gdbt_weighted, result_file_gdbt_weighted)

    # 以读二进制的方式打开文件
    file = open("./model_lr.pickle", "rb")
    # 把模型从文件中读取出来
    lr_model = pickle.load(file)
    # 关闭文件
    file.close()

    y_pred_lr = lr_model.predict(X)
    y_pred_proba_lr = lr_model.predict_proba(X)[:,1]
    result_file_lr = './lr.txt'
    save_results(GeoID, y_pred_lr, y_pred_proba_lr, result_file_lr)

#!/usr/bin/env python
# coding: utf-8

# # 使用1维卷积做滑坡敏感性制图（万州）

# In[32]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score,f1_score,precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.utils import plot_model,print_summary
from keras.layers import Conv1D,MaxPooling2D,Flatten,Softmax,Activation,Dense
import matplotlib.pyplot as plt


# In[2]:


data=np.genfromtxt('wanzhouLandslides.csv', delimiter=',',skip_header=1)


# In[3]:


print(data.shape)


# In[4]:


# 29列特征+1列标签+1个GeoID
X = data[:,0:29]
Y = data[:,29]
# print(X[0])
# print(Y[0:10])


# In[7]:


ss=StratifiedShuffleSplit(n_splits=2,test_size=0.3,random_state=23)#分成2组，测试比例为0.3

for train_index, test_index in ss.split(X, Y):
#     print("TRAIN_INDEX:", train_index, "TEST_INDEX:", test_index)#获得索引值
    x_train, x_test = X[train_index], X[test_index]#输入特征分割成2部分，训练部分和测试部分
    y_train, y_test = Y[train_index], Y[test_index]#类别集分割成2部分，训练部分和测试部分
#     print("x_train:",x_train)
#     print("y_train:",y_train)


# In[8]:


print(y_train.shape)
print(y_test.shape)
print(x_train.shape)
print(x_test.shape)


# In[10]:


# standardize train features
scaler = StandardScaler().fit(x_train)
scaled_train = scaler.transform(x_train)
scaler_test = StandardScaler().fit(x_test)
scaled_test = scaler_test.transform(x_test)

# 特征数量
nb_features = 29
# 2个类别: 滑坡与非滑坡
nb_class = 2

# reshape train data
X_train_r = np.zeros((len(x_train), nb_features, 1))
X_train_r[:, :, 0] = scaled_train[:, :nb_features]

# reshape test data
X_test_r = np.zeros((len(x_test), nb_features, 1))
X_test_r[:, :, 0] = scaled_test[:, :nb_features]

y_train_labels = np_utils.to_categorical(y_train, nb_class)
y_test_labels=np_utils.to_categorical(y_test,nb_class)


# In[11]:


def plotROC(y_test_pred, y_test_label):
    Y_pred_0 = [y[1] for y in y_test_pred]  # 取出y中的一列
    Y_test_0 = [y[1] for y in y_test_label]
    fpr, tpr, thresholds_keras = roc_curve(Y_test_0, Y_pred_0)   
    auc_val = auc(fpr, tpr)
    print("AUC : ", auc_val)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.3f})'.format(auc_val))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    #plt.savefig('results/ROC-curve.png', dpi=120)
    plt.show()


# ## case 1: conv1d->dropout->3个全连接

# In[17]:


def case1(nb_features,X_train_r,y_train_labels, nb_epoch, nb_class=2 ):
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=1, input_shape=(nb_features, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

#     sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

    nb_epoch = 10
    model.fit(X_train_r, y_train_labels, epochs=nb_epoch, batch_size=100)
    return model

m1 = case1(nb_features,X_train_r,y_train_labels,nb_epoch=20, nb_class=2 )


# In[18]:


m1_result= m1.evaluate(X_train_r,y_train_labels,batch_size=100)
print('\nTRAIN ACC :',m1_result[1])
m1_result_test= m1.evaluate(X_test_r,y_test_labels,batch_size=1000)
print('\nTEST ACC :',m1_result_test[1])
print('\n-------------------------------\n')
m1_y_test_pred = m1.predict(X_test_r, batch_size=1000)
plotROC(m1_y_test_pred, y_test_labels)


# In[20]:


#保存模型
m1.save('wanzhouM1.h5')


# ## case 2: 8个conv1d(kernel size =3 ) ->dropout->2个全连接

# In[12]:


def case2(nb_features,X_train_r,y_train_labels, nb_epoch, nb_class=2 ):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, input_shape=(nb_features, 1)))
    model.add(Activation('relu'))
    for i in range(0,7):
        model.add(Conv1D(filters=64, kernel_size=3))
        model.add(Activation('relu'))
    model.add(Conv1D(filters=32, kernel_size=1))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

#     sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(X_train_r, y_train_labels, epochs=nb_epoch, batch_size=1000)
    return model

m2 = case2(nb_features,X_train_r,y_train_labels,nb_epoch=20, nb_class=2 )


# In[13]:


result= m2.evaluate(X_train_r,y_train_labels,batch_size=100)
print('\nTRAIN ACC :',result[1])
result_test= m2.evaluate(X_test_r,y_test_labels,batch_size=1000)
print('\nTEST ACC :',result_test[1])
print('\n-------------------------------\n')
y_test_pred = m2.predict(X_test_r, batch_size=100)
plotROC(y_test_pred, y_test_labels)


# In[21]:


#保存模型
m2.save('wanzhouM2.h5')


# In[1]:


# 绘制模型
# 请参考：https://github.com/Qinbf/plot_model
print_summary(m2)
# plot_model(m2)


# In[ ]:





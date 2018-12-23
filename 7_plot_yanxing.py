#coding:utf-8

import matplotlib.pyplot as plt

scale = [14.5,6.42,7.58,5.72,4.23,4.29,2.75,9.25,6.83,37.82,0,0.61]

yanxing = ['J3s','T3xj','J3p','T2b','T1j','J1z','J1-2z','J2x','J2xs','J2s','P2','T1d']

plt.xlabel('地层岩性')
plt.ylabel('面积分布比例%')

plt.rcParams['font.sans-serif'] = ['SimHei']

plt.bar(yanxing,scale)
plt.show()

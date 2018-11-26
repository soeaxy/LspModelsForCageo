# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
 
# name_list = ['LR','GBDT','weighted GBDT']
name_list = ['Very Low','Low','Moderate', 'High', 'Very High']

LR = [0.595989389,
0.212071769,
0.116448451,
0.060098316,
0.015392076
]
GBDT = [0.822736643,
0.097684537,
0.038487723,
0.025758002,
0.015333095
]
WGBDT = [0.54459793,
0.141147543,
0.095570253,
0.08339387,
0.135290404
]
x =list(range(len(LR)))
total_width, n = 0.6, 3
width = total_width / n
 
plt.bar(x, LR, width=width, label='LR',fc = 'g')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, GBDT, width=width, label='GBDT',tick_label = name_list,fc = 'b')

for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, WGBDT, width=width, label='weighted GBDT',tick_label = name_list,fc = 'r')
plt.legend()
plt.show()

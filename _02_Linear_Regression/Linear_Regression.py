# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os
from sklearn.linear_model import Lasso

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    alpha=0.1
    ridge=Ridge(alpha=alpha,fit_intercept=True,solver='svd')
    ridge.fit(x,y)
    
def lasso(data):
  #lasso线性回归
  lasso_reg=Lasso(alpha=0.1)
  lasso_reg.fit(x,y)

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
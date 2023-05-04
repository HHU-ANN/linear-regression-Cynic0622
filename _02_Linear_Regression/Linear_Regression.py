# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os


try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y=read_data()
    XTx=x.T.dot(x)
    I_p=np.identity(x.shape[1])
    I_p[0,0]=0
    XTx+=0.5*I_p
    XTy=x.T.dot(y)
    w=np.linalg.solve(XTx,XTy)
    return np.sum(w*data)
def lasso(data):
    alpha=0.1
    num_iters=1000
    x,y=read_data()
    m, n = x.shape
    theta = np.zeros(n)
    for i in range(num_iters):
        predictions = np.dot(x, theta)
        errors = predictions - y
        gradient = np.dot(x.T, errors)
        theta -= alpha * (gradient + np.sign(theta))
    return theta

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
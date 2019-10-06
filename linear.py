import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from scipy import stats
def simpleregression(trainfile,testfile,output,weight):
    train = pd.read_csv(trainfile,header=None)
    test = pd.read_csv(testfile,header=None)
    X_train = train.values
    Y_train = X_train[:,X_train.shape[1]-1].copy()
    X_train = X_train[:,0:X_train.shape[1]-1]
    X_train = np.insert(X_train,0,1,axis = 1)
    X_test = test.values
    X_test = np.insert(X_test,0,1,axis = 1)
    w = np.linalg.pinv(X_train) @ Y_train
    for param in w:
        print(param,file=open(weight, "a"))
    pred = X_test @ w
    for pr in pred:
        print(pr,file=open(output, "a"))
def ridgeregression(trainfile,testfile,regul,output,weight):
    train = pd.read_csv(trainfile,header=None)
    test = pd.read_csv(testfile,header=None)
    X_train = train.values
    Y_train = X_train[:,X_train.shape[1]-1].copy()
    X_train = X_train[:,0:X_train.shape[1]-1]
    X_train = np.insert(X_train,0,1,axis = 1)
    X_test = test.values
    X_test = np.insert(X_test,0,1,axis = 1)
    hparams = []
    f = open(regul)
    bestpara = 0.001
    minsum = float('inf')
    r,c = X_train.shape
    fold_size = r//10
    for word in f.read().split():
        hparams.append(float(word))
    for para in hparams:
        sum1 = 0
        for i in range(10):
            t = X_train[0:i*fold_size,:]
            t1 = Y_train[0:i*fold_size]
            u = X_train[(i+1)*fold_size:r]
            u1 = Y_train[(i+1)*fold_size:r]
            X_train1 = np.concatenate((t,u),axis = 0)
            Y_train1 = np.concatenate((t1,u1),axis = 0)
            X_test1 = X_train[i*fold_size:min(((i+1)*fold_size),r),:]
            Y_test1 = Y_train[i*fold_size:min(((i+1)*fold_size),r)]
            w = np.dot(X_train1.T,X_train1)
            w = np.linalg.inv(w + para*np.eye(c))
            w = w @ (X_train1.T) @ Y_train1
            pred = np.dot(X_test1,w)
            sum1 = sum1 + np.dot((Y_test1-pred).T,(Y_test1-pred))/(10*sum(Y_test1**2))
        if sum1 < minsum:
            minsum = sum1
            bestpara = para
    print(bestpara)
    w = X_train.T @ X_train
    w = np.linalg.inv(w + bestpara*np.eye(c))
    w = w @ (X_train.T) @ Y_train
    pred = X_test @ w
    for param in w:
        print(param,file=open(weight, "a"))
    for pr in pred:
        print(pr,file=open(output, "a"))
def lassoregression(trainfile,testfile,output):
    train = pd.read_csv(trainfile,header=None)
    test = pd.read_csv(testfile,header=None)
    X_train = train.values
    # z = np.abs(stats.zscore(X_train[:,0:X_train.shape[1]-1]))
    X_train10 = X_train.copy()
    # X_train = X_train[(z < 6).all(axis=1)]
    Y_train = X_train[:,X_train.shape[1]-1].copy()
    X_train = X_train[:,0:X_train.shape[1]-1]
    X_train_numeric = np.concatenate((X_train[:,0:54],X_train[:,242:245]),axis=1).copy()
    #X_train_numeric = preprocessing.normalize(X_train_numeric)
    X_train_binary = X_train[:,54:242].copy()
    X_train_numeric1 = X_train_numeric.copy()
    # X_train_positive = np.concatenate((X_train_numeric[:,0:20],X_train_numeric[21]:X_train_numeric.shape[1]]),axis=1)
    # X_train_positive = np.log10(1 + X_train_positive)
    X_train_binary1 = X_train_binary.copy()
    #
    for i in range(2,7):
        X1 = X_train_numeric1**i
        X_train_numeric = np.concatenate((X_train_numeric,X1),axis=1)
    X_train_numeric = np.concatenate((X_train_numeric,(1/(1+np.exp(-X_train_numeric1)))),axis=1)
    X_train_numeric = np.concatenate((X_train_numeric,np.exp(-np.absolute(X_train_numeric1))),axis=1)
    # X_train_numeric = np.concatenate((X_train_numeric,np.log10(1+np.absolute(1+X_numeric1))),axis=1)
    #X_train = np.concatenate((X_train,X_train[:,0]*X_train[:,1]),axis=1)
    #X_train_numeric = np.concatenate((X_train_numeric,np.log10(1+X_train_numeric1)),axis=1)
    X_train_binary = np.concatenate((X_train_binary1,np.exp(X_train_binary1)),axis=1)
    X_train_binary = np.concatenate((X_train_binary,(1/(1+np.exp(-X_train_binary1)))),axis=1)
    X_train_binary = np.concatenate((X_train_binary,np.exp(-(X_train_binary1**2))),axis=1)
   # X_train_binary = np.concatenate((X_train_binary,(X_train_binary1<0.5)),axis=1)
    X_train = np.concatenate((X_train_numeric,X_train_binary),axis=1)
    
   # X_train = preprocessing.normalize(X_train)
    #X_train = preprocessing.scale(X_train)
    X_train = np.insert(X_train,0,1,axis = 1)
    fold_size = X_train.shape[0]//10
    r,c = X_train.shape
    X_test = test.values
    X_test_numeric = np.concatenate((X_test[:,0:54],X_test[:,242:245]),axis=1).copy()
    #X_test_numeric = preprocessing.normalize(X_test_numeric)
    X_test_binary = X_test[:,54:242].copy()
    X_test_numeric1 = X_test_numeric.copy()
    
    X_test_binary1 = X_test_binary.copy()
    for i in range(2,7):
        X2 = X_test_numeric1**i
        X_test_numeric = np.concatenate((X_test_numeric,X2),axis=1)
     
    X_test_numeric = np.concatenate((X_test_numeric,(1/(1+np.exp(-X_test_numeric1)))),axis=1)
    X_test_numeric = np.concatenate((X_test_numeric,np.exp(-np.absolute(X_test_numeric1))),axis=1)
    #X_test_numeric = np.concatenate((X_test_numeric,np.log10(1+X_test_numeric1)),axis=1)
    X_test_binary = np.concatenate((X_test_binary1,np.exp(X_test_binary1)),axis=1)
    X_test_binary = np.concatenate((X_test_binary,(1/(1+np.exp(-X_test_binary1)))),axis=1)
    X_test_binary = np.concatenate((X_test_binary,np.exp(-(X_test_binary1**2))),axis=1)
    #X_test_binary = np.concatenate((X_test_binary,(X_test_binary1<0.5)),axis=1)
    X_test = np.concatenate((X_test_numeric,X_test_binary),axis=1)
    #X_test = np.concatenate((X_test,X_test[:,0]*X_test[:,1]),axis=1)
   # X_test = preprocessing.normalize(X_test)
    #X_test = preprocessing.scale(X_test)
    X_test = np.insert(X_test,0,1,axis = 1)
    hparams = np.array([0.0001,0.001,0.003, 0.03, 1])
    minsum = float('inf')
    bestpara = 0.003
    for para in hparams:
        sum1 = 0
        for i in range(10):
            t = X_train[0:i*fold_size,:]
            t1 = Y_train[0:i*fold_size]
            u = X_train[(i+1)*fold_size:r]
            u1 = Y_train[(i+1)*fold_size:r]
            X_train1 = np.concatenate((t,u),axis = 0)
            Y_train1 = np.concatenate((t1,u1),axis = 0)
            X_test1 = X_train[i*fold_size:min(((i+1)*fold_size),r),:]
            Y_test1 = Y_train[i*fold_size:min(((i+1)*fold_size),r)]
            reg = linear_model.LassoLars(alpha=para,max_iter = 2000, normalize = True)
            reg.fit(X_train1,Y_train1)
            pred = reg.predict(X_test1)
            pred = pred*(pred>0)
            sum1 = sum1 + np.dot((Y_test1-pred).T,(Y_test1-pred))/(10*sum(Y_test1**2))
        #print(para,sum1)
        if sum1 < minsum:
            minsum = sum1
            bestpara = para
    print(bestpara)
    reg1 = linear_model.LassoLars(alpha=bestpara,fit_intercept = True, max_iter = 2000, normalize = True)
    reg1.fit(X_train,Y_train)
    w = reg1.coef_
    pred = reg1.predict(X_test)
    pred = pred * (pred>0)
    for pr in pred:
        print(pr,file=open(output, "a"))
    
if __name__ == '__main__':
    if sys.argv[1] == 'a':
        simpleregression(*sys.argv[2:])
    elif sys.argv[1] == 'b':
        ridgeregression(*sys.argv[2:])
    elif sys.argv[1] == 'c':
        lassoregression(*sys.argv[2:])
import sys
import pandas as pd
import numpy as np
import timeit
import math
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
def computeCost(X_train,Y_train,A):
    return -1/(2*X_train.shape[0]) * np.sum(Y_train * np.log(A))
def computeSoft(X_train,theta):
    A = X_train @ theta
    A = np.exp(A)
    sum1 = np.sum(A,axis=1).reshape(A.shape[0],1)
    return A/sum1

def graddesc(train, test, param, out, weight):
    train = pd.read_csv(train,header=None)
    test = pd.read_csv(test,header=None)
    X = train.values
    #Y = X[:,X.shape[1]-1].copy()
    X10 = test.values
    #Y10 = X10[:,X10.shape[1]-1].copy()
    A1 = {}
    hparams1 = []
    hparams = []
    for i in range(9):
        A1[i] ={}
    A1[0]['usual'] = 0;
    A1[0]['pretentious'] = 1;
    A1[0]['great_pret'] = 2;
    A1[1]['proper'] = 3;
    A1[1]['less_proper'] = 4;
    A1[1]['improper'] = 5;
    A1[1]['critical'] = 6;
    A1[1]['very_crit'] = 7;
    A1[2]['complete'] = 8;
    A1[2]['completed'] = 9;
    A1[2]['incomplete'] = 10;
    A1[2]['foster'] = 11;
    #A1[3]['0'] = 12;
    A1[3]['1'] = 12;
    A1[3]['2'] = 13;
    A1[3]['3'] = 14;
    A1[3]['more'] = 15;
    A1[4]['convenient'] = 16;
    A1[4]['less_conv'] = 17;
    A1[4]['critical'] = 18;
    A1[5]['convenient'] = 19;
    A1[5]['inconv'] = 20;
    A1[6]['nonprob'] = 21;
    A1[6]['slightly_prob'] = 22;
    A1[6]['problematic'] = 23;
    A1[7]['recommended'] = 24;
    A1[7]['priority'] = 25;
    A1[7]['not_recom'] = 26;
    A1[8]['not_recom'] = 0;
    A1[8]['recommend'] = 1;
    A1[8]['very_recom'] = 2;
    A1[8]['priority'] = 3;
    A1[8]['spec_prior'] = 4;
    X_train = np.zeros((X.shape[0],27))
    Y_train = np.zeros((X.shape[0],5))
    X_test = np.zeros((X10.shape[0],27))
    #Y_test = np.zeros((X10.shape[0],5))
#     E = np.concatenate((X[:,0:3],X[:,4:8]),axis=1)
#     E10 = np.concatenate((X10[:,0:3],X10[:,4:8]),axis=1)
    for i in range(X_train.shape[0]):
        for j in range(8):
            X_train[i,A1[j][X[i,j]]] = 1
#         if X[i,3] == 'more':
#             X_train[i,12] = 4
#         else:
#             X_train[i,12] = X[i,3]
        Y_train[i,A1[8][X[i,8]]] = 1   
    X_train = np.insert(X_train,0,1,axis = 1)
    for i in range(X_test.shape[0]):
        for j in range(8):
            X_test[i,A1[j][X10[i,j]]] = 1
#         if X10[i,3] == 'more':
#             X_test[i,12] = 4
#         else:
#             X_test[i,12] = X10[i,3]
#         if X10[i,3] == 'more':
#             X_test[i,12] = 4
#         else:
#             X_test[i,12] = X10[i,3]
       # Y_test[i,A1[8][X[i,8]]] = 1
    #X_train = np.insert(X_train,0,1,axis = 1)
    X_test = np.insert(X_test,0,1,axis = 1)
    #CAprint(X_train.shape)
#     enc = OneHotEncoder(handle_unknown='ignore')
#     X12 = enc.fit_transform(X).toarray()
#     X_train = X12[:,0:27].copy()
    
#     Y_train = X12[:,27:32]
#     X2 = enc.fit_transform(X10).toarray()
#     X_test = X2[:,0:27].copy()
    
#     X_test = np.insert(X_test,0,1,axis = 1)
    theta = np.zeros((X_train.shape[1],Y_train.shape[1]))
    f = open(param) 
    for word in f.read().split(', '):
        hparams1.append((word))
    for word in hparams1:
        for w in word.split():
            hparams.append(float(w))
    cost=[]
    cost.append(1000000)
    print(hparams)
    if hparams[0]==1:
        for i in range(int(hparams[2])):
#             A = X_train @ theta
#             A = np.exp(A)
            
#             #start = timeit.default_timer()
#             sum1 = np.sum(A,axis=1).reshape(A.shape[0],1)
#             A = A/sum1
            #stop = timeit.default_timer()
            #print(stop-start)
            cost1 =0
            for j in range(X_train.shape[0]//int(hparams[3])):
                X_train1 = X_train[j*int(hparams[3]):(j+1)*int(hparams[3]),:]
                Y_train1 = Y_train[j*int(hparams[3]):(j+1)*int(hparams[3])]
                A = computeSoft(X_train1,theta)
                cost1 = cost1 + ( computeCost(X_train1,Y_train1,A))
                gradient = X_train1.T @ (A - Y_train1)
                theta = theta - (1/(hparams[3]))*hparams[1]*gradient
            cost.append(cost1/j)
            if i%500==0:
                print(cost[i+1])
        #print(theta.shape)
        pd.DataFrame(theta).to_csv(weight,header=None,index=None)
        pred = np.exp(X_test @ theta)/np.sum(np.exp(X_test @ theta),axis=1).reshape(X_test.shape[0],1)
        pred2 = np.exp(X_train @ theta)/np.sum(np.exp(X_train @ theta),axis=1).reshape(X_train.shape[0],1) 
        pred2 = np.argmax(pred2,axis=1)
       # print(pred[89,:])
        pred = np.argmax(pred,axis=1)
        pred1 = []
        for i in range(pred.shape[0]):
            if pred[i] == 0:
                pred1 .append('not_recom')
            elif pred[i] == 1:
                pred1 .append('recommend')
            elif pred[i] == 2:
                pred1 .append('very_recom')
            elif pred[i] == 3:
                pred1.append('priority')
            elif pred[i] == 4:
                pred1.append('spec_prior')
        np.savetxt(out,[p for p in pred1],delimiter=',',fmt='%s')
        plt.plot(range(1,int(hparams[2]+1)),cost[1:int(hparams[2]+1)])
        plt.show()
        return (pred2,np.argmax(Y_train,axis=1))
    if hparams[0] == 2:
        cost2 =[]
        cost2.append(100000)
        for i in range(int(hparams[2])):
#             A = X_train @ theta
#             A = np.exp(A)
#             sum1 = np.sum(A,axis=1).reshape(A.shape[0],1)
#             A = A/sum1
            cost1 = 0
            for j in range(X_train.shape[0]//int(hparams[3])):
                X_train1 = X_train[j*int(hparams[3]):(j+1)*int(hparams[3]),:]
                Y_train1 = Y_train[j*int(hparams[3]):(j+1)*int(hparams[3])]
                A = computeSoft(X_train1,theta)
                cost = computeCost(X_train1,Y_train1,A)
                cost1 = cost1 + cost
                gradient = X_train1.T @ (A - Y_train1)
                theta = theta - (hparams[1]/math.sqrt(i+1))*gradient
            cost2.append(cost1/j)
            if i%500==0:
                print(cost2[i+1])
        pd.DataFrame(theta).to_csv(weight,header=None,index=None)
        pred = np.exp(X_test @ theta)/np.sum(np.exp(X_test @ theta),axis=1).reshape(X_test.shape[0],1)
        pred2 =np.exp(X_train @ theta)/np.sum(np.exp(X_train @ theta),axis=1).reshape(X_train.shape[0],1) 
        pred = np.argmax(pred,axis=1)
        pred2 = np.argmax(pred2,axis=1)
        pred1 = []
        for i in range(pred.shape[0]):
            if pred[i] == 0:
                pred1 .append('not_recom')
            elif pred[i] == 1:
                pred1 .append('recommend')
            elif pred[i] == 2:
                pred1 .append('very_recom')
            elif pred[i] == 3:
                pred1.append('priority')
            elif pred[i] == 4:
                pred1.append('spec_prior')
        plt.plot(range(1,int(hparams[2]+1)),cost2[1:int(hparams[2]+1)])
        plt.show()
        np.savetxt(out,[p for p in pred1],delimiter=',',fmt='%s')
        return pred2,np.argmax(Y_train,axis=1)
    if hparams[0] == 3:
        print(hparams)
        cost3 =[]
        cost3.append(100000)
        for i in range(int(hparams[3])):
    #             A = X_train @ theta
    #             A = np.exp(A)
    #             sum1 = np.sum(A,axis=1).reshape(A.shape[0],1)
    #             A = A/sum1
            cost1 = 0
            for j in range(X_train.shape[0]//int(hparams[4])):
                X_train1 = X_train[j*int(hparams[4]):(j+1)*int(hparams[4]),:]
                Y_train1 = Y_train[j*int(hparams[4]):(j+1)*int(hparams[4])]
                A = computeSoft(X_train1,theta)
                cost = computeCost(X_train1,Y_train1,A)
                t=1
                cost1 = cost1 + cost
                gradient = (1/X_train.shape[0])*X_train1.T @ (A - Y_train1)
                A2 = computeSoft(X_train1,theta - t*gradient)
                cost2 = computeCost(X_train1,Y_train1,A2)
                while cost2 > cost - hparams[1]*t*(np.linalg.norm(gradient))**2:
                    t = hparams[2]*t
                    A2 = computeSoft(X_train1,theta - t*gradient)
                    cost2 = computeCost(X_train1,Y_train1,A2)
                   
                theta = theta - t*gradient
            cost3.append(cost1/j)
            if i%500==0:
                print(cost3[i+1])
        pd.DataFrame(theta).to_csv(weight,header=None,index=None)
        pred = np.exp(X_test @ theta)/np.sum(np.exp(X_test @ theta),axis=1).reshape(X_test.shape[0],1)
        print(pred)
        pred2 = np.exp(X_train @ theta)/np.sum(np.exp(X_train @ theta),axis=1).reshape(X_train.shape[0],1)
        pred = np.argmax(pred,axis=1)
        pred2 = np.argmax(pred2,axis=1)
        pred1 = []
        for i in range(pred.shape[0]):
            if pred[i] == 0:
                pred1 .append('not_recom')
            elif pred[i] == 1:
                pred1 .append('recommend')
            elif pred[i] == 2:
                pred1 .append('very_recom')
            elif pred[i] == 3:
                pred1.append('priority')
            elif pred[i] == 4:
                pred1.append('spec_prior')
        plt.plot(range(1,int(hparams[3]+1)),cost3[1:int(hparams[3]+1)])
        plt.show()
        np.savetxt(out,[p for p in pred1],delimiter=',',fmt='%s')
        return pred2,np.argmax(Y_train,axis=1)
if __name__ == '__main__':
    pred,pred1 = graddesc(*sys.argv[1:])
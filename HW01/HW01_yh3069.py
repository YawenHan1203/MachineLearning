#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import svd


#load datasets 
X_test = pd.read_csv("X_test.csv",header=None,names=["w1","w2","w3","w4","w5","w6","w0"])
X_train = pd.read_csv("X_train.csv",header=None,names=["w1","w2","w3","w4","w5","w6","w0"])
y_test = pd.read_csv("y_test.csv",header=None,names=["y"])
y_train = pd.read_csv("y_train.csv",header=None,names=["y"])


# ## Part1

# ### (a)


#WRR function calculate the ridge regression coefficients
def WRR(X_train, num_lamda):   
    #get the number of features
    num_features = X_train.shape[1]
    #identity matrix
    I = np.identity(num_features)
    #transpose X_train
    X_train_trans = X_train.transpose()
    #create lamda values from 0,1,2...5000
    lamda = np.arange(num_lamda)
    #initialize the w_rr matrix
    w_rr=np.zeros((len(lamda),num_features))
    #initialize df(lamda)
    df = []
    #apply svd function for s
    _,s,_ = svd(X_train)
    #square of diagonal elements for s
    s_square = np.square(s)
    #solve for w_rr for each lamda value
    for i in lamda:
        inverse_part = inv(lamda[i]*I + np.dot(X_train_trans,X_train))
        latter_part = np.dot(X_train_trans, y_train)
        #ridge regression coefficients
        w_rr[i] = np.dot(inverse_part,latter_part).transpose()
        #solve for df(lamda)
        df_lamda = np.sum(s_square/(lamda[i]+s_square))
        df.append(df_lamda)
    return w_rr,df
    
#plot 7 values in Wrr as a dunction of df(lambda)
w_rr, df = WRR(X_train,5001)
plt.figure(1)
label = ['cylinders','displacement','horsepower','weight','acceleration','year_made','W0']
for i in range(0,7):
    plt.plot(df ,w_rr[:,i],label=label[i]) 
plt.xlabel("df($\lambda$)")
plt.ylabel("WRR")
plt.legend()
plt.title("Relationship between WRR and df($\lambda$)")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()




# ### (b)


#RMSE function calculate the rmse of prediction and actual
def RMSE(w_rr,num_lamda,X_test):
    lamda_predict = np.arange(num_lamda)
    rmse = []
    # compute root mean squared error
    for i in lamda_predict:
        y_predict = np.dot(X_test,w_rr[i])
        rmse_temp = np.sqrt(np.mean(np.square(y_test.transpose()-y_predict),axis=1))
        rmse.append(rmse_temp)
    return rmse    


#plot the RMSE on the test set as a function of lambda
rmse = RMSE(w_rr,51,X_test)
lamda_predict = np.arange(51)
plt.figure(2)
plt.plot(lamda_predict,rmse) 
plt.xlabel("$\lambda$")
plt.ylabel("RMSE")
plt.title("Root mean squared error vs. $\lambda$")
plt.show()


# ### (d)

#transform_p2 function to transform the given to p=2, then standardize
def transform_p2(X): 
    #square the given
    X2 = (X**2).iloc[:,0:6]
    #square the train
    X_train2 = (X_train**2).iloc[:,0:6]
    #rename columns
    X2.columns = [["w1^2","w2^2","w3^2","w4^2","w5^2","w6^2"]]
    X_train2.columns = [["w1^2","w2^2","w3^2","w4^2","w5^2","w6^2"]]
    #standardize using the train mean and stdev
    X2_sd = (X2-np.mean(X_train2))/np.std(X_train2)
    #combine matrix
    X_p2 = pd.concat([X, X2_sd], axis = 1)
    return X_p2
X_train_p2 = transform_p2(X_train)
X_test_p2 = transform_p2(X_test)


#transform_p3 function to transform the given to p=3, then standardize
def transform_p3(X,X_p2):
    #cube the given
    X3 = (X**3).iloc[:,0:6]
    #cube the train
    X_train3 = (X_train**3).iloc[:,0:6]
    #rename columns
    X3.columns = [["w1^3","w2^3","w3^3","w4^3","w5^3","w6^3"]]
    X_train3.columns = [["w1^3","w2^3","w3^3","w4^3","w5^3","w6^3"]]
    #standardize using the train mean and stdev
    X3_sd = (X3-np.mean(X_train3))/np.std(X_train3)
    #combine
    X_p3 = pd.concat([X_p2, X3_sd], axis = 1)
    return X_p3
X_train_p3 = transform_p3(X_train,X_train_p2)
X_test_p3 = transform_p3(X_test,X_test_p2)


#claculate rmse for p=1,2,3
w_rr1,_ = WRR(X_train,101)
rmse1 = RMSE(w_rr1,101,X_test)
w_rr2,_ = WRR(X_train_p2,101)
rmse2 = RMSE(w_rr2,101,X_test_p2)
w_rr3,_ = WRR(X_train_p3,101)
rmse3 = RMSE(w_rr3,101,X_test_p3)


#plot the test RMSE as a function of lambda=0,...,100
plt.figure(3)
label = ['p=1','p=2','p=3']
lamda_predict = np.arange(101)
plt.plot(lamda_predict,rmse1,label="p=1") 
plt.plot(lamda_predict,rmse2,label="p=2") 
plt.plot(lamda_predict,rmse3,label="p=3") 
plt.xlabel("$\lambda$")
plt.ylabel("RMSE")
plt.title("Root mean squared error vs. $\lambda$")
plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

####################
##Yawen Han
##yh3069
##HW02
##March 06, 2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import math
import seaborn as sns
from numpy.linalg import inv


#load datasets 
X = pd.read_csv("X.csv",header=None)
y = pd.read_csv("y.csv",header=None)
X = np.array(X)
y = np.array(y)


####################
##Naive Bayes

### (a)

#get the dimension of the 
num_dimension = X.shape[1]


def calculate_prob(y_train):
    '''
    calculate the pi_hat value(bernoulli distribution parameter), 
    and the probaility of spam and non-spam in y(used for posteria computation later)
    '''
    tot_records = len(y_train)
    tot_spam =  np.sum(y_train == 1)
    tot_not_spam =  np.sum(y_train == 0)
    #spam probability in y
    prob_spam = tot_spam/tot_records
    #non-spam probability in y
    prob_not_spam = tot_not_spam/tot_records 
    
    return tot_records, tot_spam, tot_not_spam, prob_spam, prob_not_spam


def calculate_lambda_hat(X_train, y_train, tot_spam, tot_not_spam):
    '''
    calculate the lambda_hat value(poisson distribution parameter), 
    '''
    dim_spam, dim_not_spam = [0]*num_dimension, [0]*num_dimension
    dim_lambda_spam, dim_lambda_not_spam = [0]*num_dimension, [0]*num_dimension 
    
    for d in range(num_dimension):    
        for i in range(len(X_train)):
            
            # Calculating total spam per dimension for spam and non-spam    
            if y_train[i] == 1.0:
                dim_spam[d] += X_train[i][d]
            elif y_train[i] == 0.0:
                dim_not_spam[d] += X_train[i][d]
        #lambda  per dimension for spam
        dim_lambda_spam[d] = (1+dim_spam[d])/(1+tot_spam)
        #lambda  per dimension for non-spam
        dim_lambda_not_spam[d] = (1+dim_not_spam[d])/(1+tot_not_spam)
    
    return dim_lambda_spam, dim_lambda_not_spam
    

def likelihood(dim_lambda_spam, X_test):
    '''
    calculate the likelihood value 
    '''
    spam_likelihood = []
    for i in range(len(X_test)):
        #calculate each term step by step 
        term1 = np.power(dim_lambda_spam,X_test[i]+1)
        
        negative_dim_lambda_spam = [-2*x for x in dim_lambda_spam]
        term2 = np.exp(negative_dim_lambda_spam)
        
        term3 =  scipy.special.factorial(X_test[i])
        
        temp1 = np.multiply(term1, term2)
        temp2 = np.divide(temp1,term3)
        temp3 = np.prod(temp2)
        #the final likelihood according to the formula
        spam_likelihood.append(temp3)
    
    return spam_likelihood
    


def predict(likelihood_spam, likelihood_not_spam, prob_spam, prob_not_spam):
    '''
    predict the posteria probability for spam and non-spam, and predict by 
    comparing their probability
    '''
    y_predict = []
    for i in range(len(likelihood_spam)):
        posteria_spam = float(prob_spam) * likelihood_spam[i]
        posteria_not_spam = float(prob_not_spam) * likelihood_not_spam[i]
        #predict y to be 1 if posteria of spam larger than non-spam
        if posteria_spam > posteria_not_spam:
            y_predict.append(1)
        else:
            y_predict.append(0)
    return y_predict




def perf_measure(y_test, y_predict):
    '''
    measure the performance by calculating the confusion matrix
    '''
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_test)): 
        if y_test[i]==1 and y_predict[i]==1:
            TP += 1
        elif y_test[i]==1 and y_predict[i]==0:
            FN += 1
        elif y_test[i]==0 and y_predict[i]==1:
            FP += 1
        elif y_test[i]==0 and y_predict[i]==0:
            TN += 1

    return TP, FP, TN, FN



from random import randrange
import random
 
# Split a dataset into k folds
def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


#initialize
tot_TP, tot_FP, tot_TN, tot_FN = 0, 0, 0, 0
lambda_spam, lambda_not_spam = [],[]

# enumerate splits
# test cross validation split
random.seed(10)
dataset = range(4600)

folds = cross_validation_split(dataset, 10)


for i in range(10):
    test_index = folds[i]
    train_index = list(set(range(4600))-set(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # use function calculate_prior to calculate tot_records, tot_spam, tot_not_spam, 
    #prob_spam, prob_not_spam, pi_hat
    tot_records, tot_spam, tot_not_spam, prob_spam, prob_not_spam = calculate_prob(y_train)
    
    # use function calculate_lambda_hat to calculate lambda_hat value per dimension for spam and non-spam
    dim_lambda_spam, dim_lambda_not_spam = calculate_lambda_hat(X_train, y_train, tot_spam, tot_not_spam)
    lambda_spam.append(dim_lambda_spam)
    lambda_not_spam.append(dim_lambda_not_spam)
    
    # use fuction likelihood to calculate likelihood value for spam and non-spam
    likelihood_spam = likelihood(dim_lambda_spam, X_test)
    likelihood_not_spam = likelihood(dim_lambda_not_spam, X_test)
    
    # use function predict to predict y
    y_predict = predict(likelihood_spam, likelihood_not_spam, prob_spam, prob_not_spam)
    
    #use fucntion perf_measure to calculate comfusion matrix
    TP, FP, TN, FN = perf_measure(y_test, y_predict)
    tot_TP += TP
    tot_FP += FP
    tot_TN += TN
    tot_FN += FN
    

# confusion matrix

cm_array = np.array([[tot_TP, tot_FN], [tot_FP, tot_TN]])
plt.figure(figsize = (6,6))
df_cm = pd.DataFrame(cm_array, index = ["True: 1","True: 0"],columns = ["Predict: 1","Predict: 0"])
_ = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d",cmap="YlGnBu",cbar_kws={"orientation": "horizontal"})
#sns.set(font_scale=4)#for label size
accuracy = (tot_TP+tot_TN)/(tot_TP+tot_TN+tot_FN+tot_FP)
_ = plt.title('Accuracy: %.2f' %accuracy )

### (b)

#the average of lambda for class "spam"(y=1) across 10 runs
lambda_spam_mean = np.array(lambda_spam).mean(0)
#the average of lambda for class "non-spam"(y=0) across 10 runs
lambda_not_spam_mean = np.array(lambda_not_spam).mean(0)


#stem plot
plt.figure(figsize = (7,6))
markerline, stemlines, baseline = plt.stem(range(1,55),lambda_spam_mean, markerfmt='o', label="$\lambda$(y=1)")
_ = plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
_ = plt.setp(stemlines, 'linestyle', 'solid')

markerline, stemlines, baseline = plt.stem(range(1,55),lambda_not_spam_mean, markerfmt='*', label="$\lambda$(y=0)")
_ = plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
_ = plt.setp(stemlines, 'linestyle', 'solid')
_ = plt.legend(loc="upper right")
_ = plt.title("Stem Plot of Poisson Parameters")
_ = plt.xlim(0,55)
_ = plt.xlabel("Dimension")
_ = plt.ylabel("$\lambda$")


##################################
#KNN

def l1_distance(instance1, instance2):
    '''
    calculate the li distance between two instances
    '''
    distance = abs(instance1-instance2).sum()
    return distance


def getNeighbors_instance(X_train, X_test_instance, k):
    '''
    find out the index of k nearest neighbours
    '''
    distances = []
    for i in range(len(X_train)):
        dist = l1_distance(X_train[i], X_test_instance)
        distances.append((i,dist))
    distances.sort(key=lambda dist: dist[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def getResponse_instance(neighbors, y_train):
    '''
    get the response for the test instance with the criteria: 
    if number of spam >number of non-spam, predict the test instance as spam(y=1)
    otherwise, predict as non-spam(y=0)
    '''
    response = []
    for i in neighbors:
        response.append(float(y_train[i]))
    return (np.mean(response)>0.5)*1
    

def getPrediction(X_train, y_train, X_test, k):
    '''
    prediction results shown as a sequence
    '''
    y_predict = []
    for i in range(len(X_test)):
        neighbors = getNeighbors_instance(X_train, X_test[i], k)
        response = getResponse_instance(neighbors, y_train)
        y_predict.append(response)
    return y_predict



def getAccuracy(y_predict, y_test):
    '''
    calculate the accuracy of the predictions
    '''
    correct = 0
    for i in range(len(y_test)):
        if y_predict[i] == y_test[i]:
             correct +=1
    accuracy = correct/float(len(y_test))
    return accuracy



from random import randrange
import random
 
# Split a dataset into k folds
def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# enumerate splits
# test cross validation split
random.seed(10)
dataset = range(4600)

folds = cross_validation_split(dataset, 10)
average_folds_accuracy = []

#knn for k =1,2,3,..20
for k in range(1,21):
    accuracy = []
    #cross-validation for 10 folds
    for i in range(10):
        test_index = folds[i]
        train_index = list(set(range(4600))-set(test_index))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # get predictions using KNN
        y_predict = getPrediction(X_train, y_train, X_test, k)
        # accuracy 
        accuracy.append(getAccuracy(y_predict, y_test))
    
    average_folds_accuracy.append(np.mean(accuracy))


fig = plt.figure(figsize=(6,5))
_ = plt.plot(range(1,21), average_folds_accuracy) 
_ = plt.xticks(range(1,21))
_ = plt.title("KNN Classifier")
_ = plt.xlabel("k")
_ = plt.ylabel("Average Accuracy")


###########################################
#Logistic Regression

#load datasets 
X = pd.read_csv("X.csv",header=None)
y = pd.read_csv("y.csv",header=None)
y[y==0] = -1 #set every y=0 to y=-1
X["intercept"] = 1 #add s dimension equal to +1 for each data points
X = np.array(X)
y = np.array(y)


def sigmoid(x):
    '''
    calculate sigmoid
    '''
    return np.exp(x)/(1.0 + np.exp(x))



def objective_fuction_value(X, y, weights):
    '''
    calculate objective function value - log liklihood
    using formula sum(sigmoid(y*X*w))
    '''
    likelihood = 0
    for i in range(X.shape[0]):
        prob = sigmoid(float(y[i])*np.dot(weights,X[i]))
        log_prob = np.log(prob)
        likelihood += log_prob  
    return log_prob


def compute_gradient(X, y, weights):
    #initialize update step
    gradient = np.zeros(X.shape[1])
    #iteration all x to update weights
    for i in range(X.shape[0]):
        prob = sigmoid(float(y[i])*np.dot(weights,X[i]))
        gradient += (1-prob)*float(y[i])*X[i]
    return gradient


def steepest_ascent(X, y, step_size):
    '''
    calculate objective function value for each iteration
    using the steepest ascent algorithm
    '''
    weights = np.zeros(X.shape[1])
    scores = []
    for step in range(1000):
        #get objective function value for each iteration
        step_score = objective_fuction_value(X, y, weights)
        scores.append(step_score)
        #call gradient function to compute gradients
        gradient = compute_gradient(X, y, weights)
        #update weights
        weights += step_size*gradient
    return scores
    

from random import randrange
import random
 
# Split a dataset into k folds
def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# enumerate splits
# test cross validation split
random.seed(1)
dataset = range(4600)

folds = cross_validation_split(dataset, 10)
step_size = 0.01/4600

fig = plt.figure(figsize=(8,8))

for i in range(10):
    test_index = folds[i]
    train_index = list(set(range(4600))-set(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #call function steepest_ascent to get the scores list for all iterations
    scores = steepest_ascent(X_train, y_train, step_size)
    #plot scores vs. iterations
    
    _ = plt.plot(range(1,1001), scores, label = "Group"+str(i+1)) 
    

_ = plt.title("Steepest Ascent: Objective vs. iteration")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Objective function value")
_ = plt.legend(loc='lower right')


### (e)

def compute_hessian(X, y, weights): 
    '''
    calculate hessian
    '''
    #initialize update step
    hessian = np.zeros((X.shape[1],X.shape[1]))
    #iteration all x to update weights
    for i in range(X.shape[0]):
        prob = sigmoid(float(y[i])*np.dot(weights,X[i]))
        hessian += prob*(1-prob)*np.matmul(np.matrix(X[i]).T,np.matrix(X[i]))
    return -hessian
    

def newtons_method(X, y, step_size):
    '''
    calculate objective function value for each iteration
    using the newtons method algorithm
    '''
    weights = np.zeros(X.shape[1])
    scores = []
    for step in range(100):
        #get objective function value for each iteration
        step_score = objective_fuction_value(X, y, weights)
        scores.append(step_score)
        #call gradient function to compute gradients
        gradient = compute_gradient(X, y, weights)
        #call hessian function to compute hessian
        hessian = compute_hessian(X, y, weights)
        #update weights
        weights -= step_size*np.matmul(inv(hessian),gradient)
    return scores, weights
    

def newtons_predict(X_test,weights):
    '''
    predictions for X_test using Newton's method
    '''
    y_predict = []
    for i in range(len(X_test)):
        prob = sigmoid(np.dot(weights,X_test[i]))
        if prob>0.5:
            predict = 1
        else:
            predict = -1
        y_predict.append(predict)
    return y_predict



def perf_measure(y_test, y_predict):
    '''
    measure the performance by calculating the confusion matrix
    '''
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_test)): 
        if y_test[i]==1 and y_predict[i]==1:
            TP += 1
        elif y_test[i]==1 and y_predict[i]==-1:
            FN += 1
        elif y_test[i]==-1 and y_predict[i]==1:
            FP += 1
        elif y_test[i]==-1 and y_predict[i]==-1:
            TN += 1

    return TP, FP, TN, FN


# enumerate splits
# test cross validation split
random.seed(1)
dataset = range(4600)
#initialize
tot_TP, tot_FP, tot_TN, tot_FN = 0, 0, 0, 0

folds = cross_validation_split(dataset, 10)
step_size = 1

fig = plt.figure(figsize=(8,8))

for i in range(10):
    test_index = folds[i]
    train_index = list(set(range(4600))-set(test_index))

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #call newton's method to get weights and objective scores
    scores, weights = newtons_method(X_train, y_train, step_size)
    #plot the scores vs. iterations
    _ = plt.plot(range(1,101), scores, label="Group"+str(i+1)) 
    #call newtons_predict to predict for test set
    y_predict = newtons_predict(X_test,weights)
    #call perf_measure to get the confusion matrix items
    TP, FP, TN, FN = perf_measure(y_test, y_predict)
    tot_TP += TP
    tot_FP += FP
    tot_TN += TN
    tot_FN += FN
    
_ = plt.title("Newton's Method :Objective vs. iteration")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Objective function value")
_ = plt.legend(loc='lower right')


# confusion matrix
cm_array = np.array([[tot_TP, tot_FN], [tot_FP, tot_TN]])
plt.figure(figsize = (6,6))
df_cm = pd.DataFrame(cm_array, index = ["True: 1","True: -1"],columns = ["Predict: 1","Predict: -1"])
_ = sns.heatmap(df_cm, annot=True, cbar=False, fmt="d",cmap="YlGnBu",cbar_kws={"orientation": "horizontal"})
#sns.set(font_scale=4)#for label size
accuracy = (tot_TP+tot_TN)/(tot_TP+tot_TN+tot_FN+tot_FP)
_ = plt.title('Accuracy: %.2f' %accuracy )





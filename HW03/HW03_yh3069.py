#!/usr/bin/env python
# coding: utf-8

# ## HW03
# 
# ### Yawen Han (yh3069)
# ### Apr 19, 2019


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings("ignore")


# ########################################################
# ## Problem 01: K-Means

# ####  Generate Data Points

# First, generate 500 observations from a mixture of three
# Gaussians on R2 with mixing weights π = [0.2, 0.5, 0.3] and means μ and covariances Σ.


## generate 500 data points
means = [[0, 0], [3, 0], [0, 3]] # means
cov = [[1, 0], [0, 1]] # covariance matrix




# the weights for mixing Gussians
weights = [0.2, 0.5, 0.3]
# generate 500 data points
data = np.zeros((500,2))
for i, weight in enumerate(weights):
    new_component = weight*np.random.multivariate_normal(means[i], cov, 500)
    data = np.add(data,new_component)
    


# #### Implement K-Means Algorithm

# Then implement K-Means algorithm, with **"num_centroids"** for number of clusters and **"iterations"** for terminal criteria.



def kmeans(data,num_centroids,iterations):
    '''
    Implement the k-Means algorithm
    '''
    dim = data.shape[1] # dimension for each centroid
    num_points = data.shape[0] # number of data points
    objective = [] # list to store the objective function values
    
    #1. A random initialization for centroids
    data_mean = np.mean(data)
    data_std = np.std(data)
    centers = np.random.randn(num_centroids,dim)*data_std + data_mean
    
    #2. Data assignment and centroids update
    clusters = np.zeros(num_points) # array to store cluster assignments
    distances = np.zeros((num_points,num_centroids)) # array to store distance from each data points to each centroid
    
    # When, after an update, finish all iterations, exit loop
    for i in range(iterations):
        
        # Measure the distance to every center
        for c in range(num_centroids):
            distances[:,c] = np.linalg.norm(data - centers[c], axis=1)
        
        # Assign all training data to closest center
        clusters = np.argmin(distances, axis = 1)
        # objective function value
        objective.append(np.sum(np.min(distances,axis=1)))
        
        # Calculate mean for every cluster and update the center
        for c in range(num_centroids):
            centers[c] = np.mean(data[clusters == c], axis=0)
            
    #return the objective function values for all iterations
    return objective, clusters, centers


# ### Question 1-a: Objective vs. Iteration for k=2,3,4,5

# For K = 2, 3, 4, 5, plot the value of the K-means objective function per iteration for 20 iterations
# 


fig = plt.figure(figsize=(8, 7))
# call the function kmeans and plot te objective value for 20 iterations
K = [2,3,4,5]
for k in K:
    objective,_,_ = kmeans(data,num_centroids=k,iterations=20)
    plt.plot(np.arange(1,21),objective,label='k={}'.format(k))
# modify plot format
plt.title("K-Means Objective value for differnet k")
plt.xlabel("Iteration")
plt.ylabel("Objective value")
_ = plt.xticks(np.arange(1,21))
_ = plt.legend()


# According to the plot above, we can conclude that with the increasing number of k(number of clusters), the objective function get a samller value. This makes sense as the more clusters, the relative distance between each data point and corresponding centroid is smaller. 

# ### Question 1-b: Cluster plots for k=3,5

# For K = 3, 5, plot the 500 data points and indicate the cluster of each for the final iteration by marking it with a color or a symbol.



# Clustering plot for 500 data with k=3
objective, clusters, centers = kmeans(data,num_centroids=3,iterations=20)
colors=['orange', 'blue', 'green']
fig = plt.figure(figsize=(8, 6))
# scatterplot for points with assigned clusters
_ = plt.scatter(data[:,0],data[:,1],color=[colors[i] for i in clusters.tolist()])
# centroids
_ = plt.scatter(centers[:,0],centers[:,1],color='red',marker="*",s=150)
_ = plt.title("Clustering: k=3")


# For **k=3**, the data points assignments and centroids are shown above.


# Clustering plot for 500 data with k=5
objective, clusters, centers = kmeans(data,num_centroids=5,iterations=20)
colors=['orange', 'blue', 'green', 'c','m']
fig = plt.figure(figsize=(8, 6))
# scatterplot for points with assigned clusters
_ = plt.scatter(data[:,0],data[:,1],color=[colors[i] for i in clusters.tolist()])
# centroids
_ = plt.scatter(centers[:,0],centers[:,1],color='red',marker="*",s=150)
_ = plt.title("Clustering: k=5")


# For **k=5**, the data points assignments and centroids are shown above. Compared to k=3, some clusters are more sparse than others.



# #############################################################
# ## Problem 02: Bayes Classifier Revisited

# In this section, the **EM** algorithm for the **Gaussian mixture** model is implemented, with the purpose of using it in a **Bayes classifier**.


# load datasets
X_test = pd.read_csv("Prob2_Xtest.csv",header=None)
X_train = pd.read_csv("Prob2_Xtrain.csv",header=None)
y_test = pd.read_csv("Prob2_ytest.csv",header=None)
y_train = pd.read_csv("Prob2_ytrain.csv",header=None)



def initialize_model(data, num_clusters):
    '''
    initialize all covariance matrices to the empirical covariance of the data being modeled. 
    Randomly initialize the means by sampling from a single multivariate Gaussian where the 
    parameters are the mean and covariance of the data being modeled. Initialize the 
    mixing weights to be uniform.
    '''
    # sample data covariance 
    sample_cov = np.cov(np.array(data).T)
    # sample data means 
    sample_mean = np.mean(data,axis=0).tolist()
    # initial means: sampling from a single multivariate Guassian with mean and cov of the data being modeled
    ini_means = np.random.multivariate_normal(sample_mean, sample_cov, num_clusters)
    # initial covariance
    ini_covs = [sample_cov for i in range(data.shape[1])]
    # initial weights
    ini_weights = np.ones(num_clusters)/num_clusters
    
    return ini_means, ini_covs, ini_weights
    


# ###  EM Algirithm

# Implement the EM algorithm by iterating E-step and M-step.

# #### 1. LogLikelihood



def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):
            
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll


# #### 2. E-Step


from scipy.stats import multivariate_normal

def compute_responsibilities(data, weights, means, covariances):
    '''E-step: compute responsibilities, given the current parameters'''
    num_data = len(data)
    num_clusters = len(means)
    resp = np.zeros((num_data, num_clusters))
    
    # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
    # Hint: To compute likelihood of seeing data point i given cluster k, use multivariate_normal.pdf.
    for i in range(num_data):
        for k in range(num_clusters):
            resp[i, k] = weights[k]*multivariate_normal.pdf(data[i],mean=means[k],cov=covariances[k])
    
    # Add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums
    
    return resp


# #### 3.M-Step



def compute_soft_counts(resp):
    # Compute the total responsibility assigned to each cluster, which will be useful when 
    # implementing M-steps below. In the lectures this is called N^{soft}
    counts = np.sum(resp, axis=0)
    return counts




def compute_weights(counts):
    num_clusters = len(counts)
    weights = [0.] * num_clusters
    
    for k in range(num_clusters):
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        # compute # of data points by summing soft counts.
        weights[k] = counts[k]/np.sum(counts)

    return weights




def compute_means(data, resp, counts):
    '''M-step: update means'''
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters
    
    for k in range(num_clusters):
        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        weighted_sum = 0.
        for i in range(num_data):
            weighted_sum += resp[i,k]*data[i]
        
        means[k] = weighted_sum/counts[k]

    return means



def compute_covariances(data, resp, counts, means):
    '''M-step: update covariances'''
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim,num_dim))] * num_clusters
    
    for k in range(num_clusters):
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            weighted_sum += resp[i,k]*np.outer(data[i]-means[k],data[i]-means[k])
        
        covariances[k] = weighted_sum/counts[k]

    return covariances


# #### 3. EM algo


def EM(data, init_means, init_covariances, init_weights, maxiter):
    '''
    Use EM algotithm for the GMM model
    '''
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for it in range(maxiter):
        #if it % 5 == 0:
        #    print("Iteration %s" % it)
        
        # E-step: compute responsibilities
        resp = compute_responsibilities(data, weights, means, covariances)

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = compute_soft_counts(resp)
        
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        weights = compute_weights(counts)
        
        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        means = compute_means(data, resp, counts)
        
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        covariances = compute_covariances(data, resp, counts, means)
        
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        #if (ll_latest - ll) < thresh and ll_latest > -np.inf:
         #   break
        ll = ll_latest
    
    #if it % 5 != 0:
        #print("Iteration %s" % it)
    
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out


# ### Question 2-a: GMM with EM

# plot the log marginal objective function for a 3-Gaussian mixture model over 10 different runs and for iterations 5 to 30.


def plot_GMM(data, num_clusters):
    # plot 
    fig = plt.figure(figsize=(8, 6))

    #initial best results
    best_l1 = 0
    best_result = {}

    #iterate over 10 different runs
    for i in range(10):
        
        # Initialize the multivariate Guassiam models 
        ini_means, ini_covs, ini_weights = initialize_model(data, num_clusters)
        # Run EM 
        results = EM(np.array(data), ini_means, ini_covs, ini_weights, 30)
        # plot
        _ = plt.plot(np.arange(5,31),results['loglik'][5:],label='#{}'.format(i+1))
        # get the best results with max objective function value
        if results['loglik'][-1] > best_l1:
            best_result = results
    
    # modify plot format
    plt.title("EM algo of GMM for 10 runs")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    _ = plt.legend()
    
    return best_result
    




# For class = 0
X_train_spam = X_train.loc[y_train[0] == 0]
best_result_spam = plot_GMM(X_train_spam, num_clusters=3)


# The plot above for **class = 0**, demonstrating the log marginal objective function for a 3-Gaussian mixture model over 10 different runs and for iterations 5 to 30.


# For class = 1
X_train_nonspam = X_train.loc[y_train[0] == 1]
best_result_nonspam = plot_GMM(X_train_nonspam, num_clusters=3)


# The plot above for **class = 1**, demonstrating the log marginal objective function for a 3-Gaussian mixture model over 10 different runs and for iterations 5 to 30.

# ### Question 2-b: Bayes Classifier

# #### 1. The best run prediction for (a)

# First, using the best run for each class after 30 iterations, predict the testing data usinga Bayes classifier and show the result in a 2 × 2 confusion matrix, along with the accuracy percentage.



def compute_conditional_density(data, best_result):
    '''compute conditional density, given the current parameters'''
    # get required paras
    weights = best_result['weights']
    means = best_result['means']
    covariances = best_result['covs']
    
    # data dimension
    num_data = len(data)
    num_clusters = len(means)
    condens = np.zeros((num_data, num_clusters))
    
    # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
    for i in range(num_data):
        for k in range(num_clusters):
            condens[i, k] = weights[k]*multivariate_normal.pdf(data[i],mean=means[k],cov=covariances[k])
    
    # Add up responsibilities over each data point and normalize
    condens_sums = condens.sum(axis=1)[:, np.newaxis]
    
    return condens_sums


def bayes_classification(best_result_spam, best_result_nonspam):
    pred = []
    
    # get the conditional density for spam and nonspam
    spam_dens = compute_conditional_density(np.array(X_test), best_result_spam)
    nonspam_dens = compute_conditional_density(np.array(X_test), best_result_nonspam)
    
    # predict the class to be the better prior*conditional density
    spam_post = spam_prior*spam_dens
    nonspam_post = nonspam_prior*nonspam_dens
    
    for i in range(len(spam_post)):
        if spam_post[i] >= nonspam_post[i]:  
            pred.append(0)
        else:
            pred.append(1)
        
    return pred
          


# get the prior for each class
spam_prior = sum((y_train[0]==0)*1)/len(y_train)
nonspam_prior = sum((y_train[0]==1)*1)/len(y_train)



import sklearn.metrics as metrics

#predict
best_pred = bayes_classification(best_result_spam, best_result_nonspam)
#accuracy for best model in (a)
best_acc = metrics.accuracy_score(y_test, best_pred)
# confusion matrix
best_cm = metrics.confusion_matrix(y_test, best_pred)

# heatmap for confusion matrix and accuracy
fig = plt.figure(figsize=(6,5))
best_heatmap = sns.heatmap(best_cm, cmap="PuBu",annot=True, fmt="d")
best_heatmap.set_ylabel("Real value")
best_heatmap.set_xlabel("Predicted value")
best_heatmap.set_title('Best {0}-Guassian with accuracy={1:.2f}%'.format(3,best_acc*100))


# The confusion matrix for the best run in (a) is shown above, the accuracy is **82.17%**.

# #### 2. Repeat the process for 1-4 Guassian Mixture Model

# Repeat this process for a 1-, 2-, 3- and 4-Gaussian mixture model.



def get_GMM_result(data, num_clusters):
    # Initialize the multivariate Guassiam models 
    ini_means, ini_covs, ini_weights = initialize_model(data, num_clusters)
    # Run EM 
    result = EM(np.array(data), ini_means, ini_covs, ini_weights, 30)
    
    return result
    



def prdict_confusion_matrix(num_clusters):

    result_spam = get_GMM_result(X_train_spam, num_clusters)
    result_nonspam = get_GMM_result(X_train_nonspam, num_clusters)
    # Predict
    y_pred = bayes_classification(result_spam, result_nonspam)
    
    return y_pred



fig = plt.figure(figsize=(12,10))
accuracy = []
#plot for 1-4 Guassian Mixture Model
for i in np.arange(1,5):
    
    y_pred = prdict_confusion_matrix(num_clusters=i)#predict
    cm = metrics.confusion_matrix(y_test, y_pred)#confusion matrix
    arr = metrics.accuracy_score(y_test, y_pred)#accuracy
    
    #heatmap for confusion matrix
    cm_heatmap = fig.add_subplot(2, 2, i)
    cm_heatmap = sns.heatmap(cm, cmap="PuBu",annot=True, fmt="d")
    cm_heatmap.set_ylabel("Real value")
    cm_heatmap.set_xlabel("Predicted value")
    cm_heatmap.set_title('{0}-Guassian with accuracy={1:.2f}%'.format(i,arr*100))
    #accuracy list
    accuracy.append(arr)
    
plt.tight_layout()


# The confusion matrixes for for a 1-, 2-, 3- and 4-Gaussian mixture model are shown above, with accuarcy shown in the title.


accuracy_perc = [str(np.round(acc*100,2))+"%" for acc in accuracy]
pd.DataFrame({'#-Guassian Mixture':np.arange(1,5),'Accuracy':accuracy_perc}).set_index('#-Guassian Mixture')


# The accuarcy for each Gussian mixture model is also summarized in the table above.



# #################################################
# ## Problem 03: Matrix Factorization

# In this section, the MAP inference algorithm for the matrix completion problem is implemented.


# load datasets
train_set = pd.read_csv("Prob3_ratings.csv",header=None,names=['user_id','movie_id','rating'])
test_set = pd.read_csv("Prob3_ratings_test.csv",header=None,names=['user_id','movie_id','rating'])


# Before the implemention, let's have a look at the dataset:


train_set.head(10)


# The dataset is summarized to exclude the missing values. In order to implement the algorithm, it will be transformed to the sparse matrix first.


def generate_Mij(df): 
    '''Trasnform the given dataframe to sparse matrix'''
    M = np.zeros((n_users,n_movies)) 
    for row in df.iterrows():
        Mrow = int(row[1]['user_id']) - 1
        Mcol = int(row[1]['movie_id']) - 1
        Mval = row[1]['rating']
        M[Mrow, Mcol] = Mval
    return M



# get the number of users of movies
n_users = np.max(train_set['user_id'])
n_movies = np.max(train_set['movie_id'])

#generate user-movie matrix
#training set
ratings = generate_Mij(train_set)
#testing set
ratings_test = generate_Mij(test_set)

print ('There are %i unique users' %n_users)
print ('There are %i unique movies' %n_movies)



# check the sparsity of the user-movie matrix
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0] * ratings.shape[1])
sparsity *= 100
print ('percentage of user-movies that have a rating: {:.2f}%'.format(sparsity))


# ### Matrix Factorization

# Implement the algorithm by optimizing the MAP, and update the u_i and v_j locations iteratively.


#P is latent user feature matrix
#Q is latent movie feature matrix
def prediction(P,Q):
    return np.dot(P.T,Q)


sig2 = 0.25 #variance
d = 10 #number of latent features
lmbda = 1 # Regularization parameter
n_iteration = 100  # Number of iterations



# initialize P and Q
mean = np.zeros(d)
cov = np.eye(d)
P = np.random.multivariate_normal(mean, cov, n_users).T
Q = np.random.multivariate_normal(mean, cov, n_movies).T

# shape of P and Q
print ('Shape of P: ', P.shape)
print ('Shape of Q: ', Q.shape)



def initialize(num_users,num_movies):
    # initialize P and Q
    mean = np.zeros(d)
    cov = np.eye(d)
    ini_P = np.random.multivariate_normal(mean, cov, num_users).T
    ini_Q = np.random.multivariate_normal(mean, cov, num_movies).T
    return ini_P,ini_Q
    



from numpy.linalg import inv

def update_location(P,Q):
    '''update the P or Q location(correspond to the argument positions)'''
    for i in range(P.shape[1]):
        # iterate all u_i
        u_i = P[:,i][:,np.newaxis]
        vj_sum = np.zeros((d,d)) # initialize the sum of v_j
        Mij_vj = np.zeros((d,1)) # initialize the sum of M_ij*v_j
        
        # get the M_ij according to update P or Q
        if P.shape[1] == n_users:
            M = ratings
        elif P.shape[1] == n_movies:
            M = ratings.T
    
        # iterate all nonzero M_ij to update u_i
        for j in M[i,].nonzero()[0]:
           
            v_j = Q[:,j][:,np.newaxis]
            #update the sum of v_j and M_ij*v_j
            vj_sum = np.add(vj_sum, np.dot(v_j,v_j.T) )
            Mij_vj = np.add(Mij_vj, M[i,j]*v_j)
            
        # the algorithm formula 
        inverse_part = inv(np.add(lmbda*sig2*np.eye(d), vj_sum))
      
        u_i = np.dot(inverse_part, Mij_vj)
        # update each u_i
        P[:,i] = u_i.T[0]
        
    return P
        


def map_inference(num_users,num_movies,num_iteration,M):
    obj_val = []
    #initialize P, Q
    P, Q = initialize(num_users,num_movies)
    #100 iterations to update P, Q
    for iteration in range(num_iteration):
        # update P
        P = update_location(P,Q)
        # update Q
        Q = update_location(Q,P)
        # predict
        predict = prediction(P,Q)
        # get objective function value
        objective_val = get_object_val(P,Q,M,predict)
        obj_val.append(objective_val)
    return P,Q,obj_val
    



def get_object_val(P,Q,M,pred):
    # lambda/2/sig^2*(M_ij-u_i*v_j)^2
    first_comp = lmbda/2/sig2*np.square(np.subtract(M,pred))
    # get the index of nonzero value in M_ij
    nonzero_idx = M.nonzero()
    # -1*sum(first_comp in nonzero index)
    first_comp_sum = -1*np.sum(first_comp[nonzero_idx])
    # -1*sum(lambda/2*(u_i)^2)
    second_comp_sum = -1*np.sum(lmbda/2*np.square(P))
    # -1*sum(lambda/2*(v_j)^2)
    third_comp_sum = -1*np.sum(lmbda/2*np.square(Q))
    #objective function = fist_comp_sum+second_comp_sum+third_comp_sum
    object_val = first_comp_sum+second_comp_sum+third_comp_sum
    
    return object_val
    


def get_rmse(M_test,pred):
    #get the nonzero index for testing set
    test_nonzero_idx = M_test.nonzero()
    # true value
    real_ratings=M_test[test_nonzero_idx]
    # prediction value
    pred_ratings=pred[test_nonzero_idx]
    # root mean squred error
    rmse = np.linalg.norm(pred_ratings-real_ratings)/np.sqrt(len(pred_ratings))
    
    return rmse
    


# ### Question 3-a: MAP Inference Algo for Matrix Factorization

# Run your code 10 times. For each run, initialize your ui and vj vectors as N (0, I ) random vectors.



# run the MAP Inference Algo for 10 times
fig = plt.figure(figsize=(8, 6))
rmse_ls = [] #list store rmse for 10 runs
obj_val_ls = [] #list store final obj val for 10 runs
best_obj_val = float("-inf")#initial the best objective value

for i in range(10):
    # get the updated P,Q after 100 iterarions    
    U, V, obj_val = map_inference(n_users,n_movies,n_iteration,ratings)
    # predictions
    final_pred = prediction(U,V)
    # get the final objective function value
    final_obj_val = get_object_val(U,V,ratings,final_pred)
    obj_val_ls.append(final_obj_val)
    # get the RMSE
    rmse = get_rmse(ratings_test,final_pred)
    rmse_ls.append(rmse)
    
    #get the best U,V for best obj fun
    if final_obj_val>best_obj_val:
        best_obj_val = final_obj_val
        best_U = U
        best_V = V
    
    #plot
    _ = plt.plot(np.arange(1,100),obj_val[1:],label='#{}'.format(i+1))
    
# modify plot format
plt.title("MAP Inference Algo for 10 runs")
plt.xlabel("Iteration")
plt.ylabel("Objective value")
_ = plt.legend()


# On the plot above, it shows the the log joint likelihood for iterations 2 to 100 for each run. The objective function value is improved after 100 iterations, and reach the stable stage.


df_summary = pd.DataFrame({'#run':np.arange(1,11),'Objective_val':obj_val_ls,'RMSE':rmse_ls})
df_summary.sort_values(by=['Objective_val'],ascending=False)


# In the table above, it show in each row the final value of the training objective function next to the RMSE on the testing set. Sort these rows according to decreasing value of the objective function.

# ### Question 3-b: Find Closest Movies

# For the run with the highest objective value, pick the movies **“Star Wars”**, **“My Fair Lady”** and **“Goodfellas”** and for each movie find the 10 closest movies according to Euclidean distance using their respective locations vj . List the query movie, the ten nearest movies and their distances

# load datasets
movies = pd.read_csv("Prob3_movies.txt",header=None,names=['Name'])
# list of movie names to find closest movies
movie_names =["Star Wars","My Fair Lady","GoodFellas"]


def find_idx(name):  
    '''Find the index of the movie with the given name'''
    idx = movies[movies["Name"].str.contains(name, na=False)].index
    return idx



def get_euclidean_dist(matrix,vector):
    # calculate the euclidean distance between matrix and vector
    distances = np.sqrt(np.sum(np.square(matrix-vector),axis=0))
    return distances



def get_10_closest_movie(movieName):
    # get the index of the movie
    movie_idx = find_idx(movieName)
    # get the vector of the movie
    movie_vec = best_V[:,movie_idx]
    # get the euclidean distance 
    movies['dist']=get_euclidean_dist(best_V,movie_vec)
    # sort movies
    top_10 = movies.sort_values(by=['dist'])[1:11]
    
    return top_10
    


# STAR WARS
print("Top10 closest movies for",movie_names[0],":\n")
get_10_closest_movie(movie_names[0])



# MY FAIR LADY
print("Top10 closest movies for",movie_names[1],":\n")
get_10_closest_movie(movie_names[1])


# GOODFELLAS
print("Top10 closest movies for",movie_names[2],":\n")
get_10_closest_movie(movie_names[2])



import IPython.core.display as di
# This line will add a button to toggle visibility of code blocks, for use with the HTML export version
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)






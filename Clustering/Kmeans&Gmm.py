
# coding: utf-8

# In[ ]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import renders as rs
from sklearn.mixture import GMM

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
get_ipython().magic(u'matplotlib inline')

# Load the wholesale customers dataset

try:
    data = pd.read_csv("data.csv")
    data = data.set_index('id')
    #data.drop('Font_id', axis = 1, inplace = True)
   
    
    print "dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


# In[ ]:

# Display a description of the dataset
display(data.describe())
Kmeans(data)


# In[1]:

#  Apply your clustering algorithm of choice to the reduced data 
def Kmeans(data):
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=20, init='k-means++', n_init=10, max_iter=300)
    clusterer =clusterer.fit(data)

    # Predict the cluster for each data point
    preds = clusterer.predict(data)

    # Find the cluster centers
    centers = clusterer.cluster_centers_
    #  Predict the cluster for each transformed sample data point

    #  Calculate the mean silhouette coefficient for the number of clusters chosen
    from sklearn.metrics import silhouette_score
    score = silhouette_score(data,preds)
    print score


# In[2]:


#gmm Clustering 

def Gmm(data):
    gaussian = GMM(n_components=10,n_iter=500,covariance_type='spherical' )
    gaussian.fit(data)
    preds2= gaussian.predict(data)
    print preds2
    from sklearn.metrics import silhouette_score
    score = silhouette_score(data,preds2)
    print score


# In[ ]:




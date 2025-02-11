#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


#  #### Clustering-Divide the universities in to group(Clusters)
# 

# In[4]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[5]:


Univ.info()


# In[6]:


np.mean(Univ["SAT"])


# In[7]:


np.median(Univ["SAT"])


# In[8]:


np.var(Univ["SFRatio"])


# In[9]:


Univ.describe()


# In[10]:


# Read all numeric columns into Univ1
Univ1 = Univ.iloc[:,1:]


# In[11]:


Univ1


# In[12]:


cols = Univ1.columns


# In[13]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df
# scaler.fit_transform(Univ1)


# In[15]:


# Build 3 clusters using KMeans Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[16]:


# Print the Cluster labels
clusters_new.labels_


# In[18]:


set(clusters_new.labels_)


# In[19]:


Univ['clusterid_new'] = clusters_new.labels_


# In[20]:


Univ


# In[21]:


Univ.sort_values(by = "clusterid_new")


# In[23]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# #### Observations

# * Cluster 2 appears to be the top rated universities cluster as the cut off score, Top 10,SFRatio parameter mean values are high
# * Cluster 1 appears to occupy the middle level rated universities
# * Cluster 0 comes as the lower level rated universities

# In[24]:


Univ[Univ['clusterid_new']==0]


# #### Finding optimal K vslue using elbow plot
# 

# In[27]:


wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    #kmeans.fit(Univ1)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# #### Observations
# * From the above graph we can choose 3 or 4 which indicates elbow joint i.e the rate of change of slop decreases

# In[ ]:





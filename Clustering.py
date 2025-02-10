#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# #### Clustering-Divide the universities in to group(Clusters)

# In[8]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[11]:


Univ.info()


# In[12]:


np.mean(Univ["SAT"])


# In[13]:


np.median(Univ["SAT"])


# In[14]:


np.var(Univ["SFRatio"])


# In[15]:


Univ.describe()


# In[19]:


# Read all numeric columns into Univ1
Univ1 = Univ.iloc[:,1:]


# In[20]:


Univ1


# In[24]:


cols = Univ1.columns


# In[27]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df
# scaler.fit_transform(Univ1)


# In[ ]:





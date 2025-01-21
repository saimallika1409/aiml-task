#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[5]:


#mean value of SAT score
np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[6]:


#find the variance
np.var(df["SFRatio"])


# In[7]:


df.describe()


# In[ ]:





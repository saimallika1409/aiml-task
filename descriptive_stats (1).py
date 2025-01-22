#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


df = pd.read_csv("Universities.csv")
df


# In[6]:


#mean value of SAT score
np.mean(df["SAT"])


# In[7]:


np.median(df["SAT"])


# In[8]:


#find the variance
np.var(df["SFRatio"])


# In[9]:


df.describe()


# ##visualizations

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


plt.hist(df["GradRate"])


# In[15]:


plt.figure(figsize=(6, 9))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# In[ ]:





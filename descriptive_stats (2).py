#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


#mean value of SAT score
np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[5]:


#find the variance
np.var(df["SFRatio"])


# In[6]:


df.describe()


# ##visualizations

# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


plt.hist(df["GradRate"])


# In[14]:


plt.figure(figsize=(6, 9))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# ##visualizations using boxplot

# In[20]:


s = [20,15,10,12,13,14,10,160,150]
scores = pd.Series(s)
scores


# In[21]:


plt.boxplot(scores, vert=False)


# In[22]:


plt.figure(figsize=(6,2))
plt.title("Box plot for SAT Score")
plt.boxplot(df["SAT"], vert = False)


# In[ ]:





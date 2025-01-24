#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[4]:


#printing the information
data.info()


# In[6]:


#dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[8]:


#drop duplocate column(temp c) and unnamed column
data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[9]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[13]:


#print all duplicated rows
data1[data1.duplicated(keep = False)]


# In[14]:


#drop duplicated rows
data1.drop_duplicates(keep='first', inplace = True)
data1


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ###import libraries and Data Set

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[5]:


sns.histplot(data1['daily'], kde = True,stat='density',)


# In[6]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["sunday"], vert = False)


# In[8]:


sns.histplot(data1['sunday'], kde = True,stat='density',)


# ##observations
# -There are no missing values
# -The daily column values appears to be right skewed
# -The sunday column values also appear to be right skewed
# there are two outliers in both daily column and also in sunday column as observed 

# Sctter plot and correlation strength

# In[10]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[12]:


data1["daily"].corr(data1["sunday"])


# In[13]:


data1[["daily","sunday"]].corr()


# In[14]:


data1.corr(numeric_only=True)


# 
# observations on correlation strength

# 
# -The relationship between x (daily) and y(sunday) is seen to be linear as seen from scatter plot
# -The correlation is strong and positive with pearson's correlation coefficient of 0.958154

# Fit a Linear Regression Model

# In[15]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[16]:


model1.summary()


# In[ ]:





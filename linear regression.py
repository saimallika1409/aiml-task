#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[19]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[8]:


data1.info()


# In[9]:


data1.isnull().sum()


# In[11]:


data1[data1.duplicated(keep = False)]


# In[12]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[13]:


data1.describe()


# In[14]:


cols = data1.columns
colors = ['black', 'yellow']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# ##observations

# -there are no missing values
# -there are no duplicate values
# 

# In[15]:


plt.hist(data1['daily'])


# In[22]:


plt.scatter(data1['daily'],data1['sunday'])


# In[26]:


sns.boxplot(data = data1, x = "daily", y="sunday")


# In[27]:


data1["daily"].corr(data1["sunday"])


# In[28]:


data1[["daily","sunday"]].corr()


# In[29]:


data1.corr(numeric_only=True)


# In[30]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[31]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[32]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[33]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
 
# plotting the regression line
plt.plot(x, y_hat, color = "g")
  
# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[34]:





# In[ ]:





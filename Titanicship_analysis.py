#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Istall mlxtend library
get_ipython().system('pip install mlxtend')


# In[5]:


# Import necessary libraries
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[7]:


# Print the dataframe
titanic = pd.read_csv("Titanic.csv")
titanic


# In[8]:


titanic.info()


# In[9]:


titanic.isnull().sum()


# In[10]:


titanic.describe()


# #### Observations
# * There are no null values
# * All columns are object datatype
# * All objects are categorical in nature
# * As the columns are categorical, we can adopt one-hot-encoding

# In[12]:


# plot a bar chat to visualize the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[13]:


titanic['Age'].value_counts()


# In[14]:


titanic['Gender'].value_counts()


# #### Observations
# * we observed that crew stands first and next 3rd and next 1st and finally last is 2nd

# In[15]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# #### observations 
# * Here we can observe that adults percentage is more than the child

# In[16]:


counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# #### observations
# * here we can see that male are more than female

# In[17]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[18]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[19]:


df.info()


# #### Apriori Algorithm

# In[20]:


frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True,max_len=None)
frequent_itemsets


# In[21]:


frequent_itemsets.info()


# In[23]:


# Generate association rules with metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[24]:


rules.sort_values(by='lift', ascending = True)


# In[25]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:





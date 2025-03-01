#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[ ]:


# import some dataset from sklearn
iris = datasets.load_iris(as_frame=True).frame


# In[ ]:


iris = pd.read_csv("iris.csv")
iris


# In[ ]:


# Bar plot for categorical column
import seaborn as sns
counts =  iris["variety"].value_counts()
sns.barplot(data = counts)


# In[ ]:


iris.info()


# In[ ]:


iris[iris.duplicated(keep=False)]


# #### Perform label encoding of target column

# In[10]:


# Encode the three flower classes as 0,1,2
labelencoder = LabelEncoder()
iris.iloc[:,-1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[11]:


iris.info()


# #### Observation
# * The target column('variety) is still object type.It needs to be converted to numeric(int)

# In[12]:


# Convert the target column data type to integer
iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[14]:


# Divide the dataset
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[15]:


Y


# In[16]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# In[17]:


x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
y_train


# #### Building Decision Tree Classifier using Entropy criteria

# In[18]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=None)
model.fit(x_train,y_train)


# In[21]:


#Plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[ ]:





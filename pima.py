#!/usr/bin/env python
# coding: utf-8

# In[16]:


pip install joblib


# In[17]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[18]:


# Load dataset
df = pd.read_csv("diabetes.csv")
df


# In[19]:


dataframe.info()


# In[20]:


dataframe = pd.read_csv("diabetes.csv")
dataframe


# In[21]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Assuming 'dataframe' is already defined
X = dataframe.iloc[:, 0:8]  # Features
Y = dataframe.iloc[:, 8]    # Target

# Setup cross-validation using StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=3, shuffle=True)

# Create RandomForestClassifier model
model = RandomForestClassifier(n_estimators=200, random_state=20, max_depth=None)

# Evaluate model using cross-validation
results = cross_val_score(model, X, Y, cv=kfold)

# Print the results of cross-validation and the mean accuracy
print(results)
print(results.mean())


# In[22]:


X = df.drop('class', axis=1)
y = df['class']


# In[23]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[24]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


# In[ ]:





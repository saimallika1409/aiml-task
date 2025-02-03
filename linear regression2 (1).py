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


# In[7]:


sns.histplot(data1['sunday'], kde = True,stat='density',)


# ##observations
# -There are no missing values
# -The daily column values appears to be right skewed
# -The sunday column values also appear to be right skewed
# there are two outliers in both daily column and also in sunday column as observed 

# Sctter plot and correlation strength

# In[8]:


x = data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[9]:


data1["daily"].corr(data1["sunday"])


# In[10]:


data1[["daily","sunday"]].corr()


# In[11]:


data1.corr(numeric_only=True)


# 
# observations on correlation strength

# 
# -The relationship between x (daily) and y(sunday) is seen to be linear as seen from scatter plot
# -The correlation is strong and positive with pearson's correlation coefficient of 0.958154

# Fit a Linear Regression Model

# In[12]:


import statsmodels.formula.api as smf
model1 = smf.ols("sunday~daily",data = data1).fit()


# In[14]:


model1.summary()


# interpretation

# -R^2 = 1 = perfect fit(all variance explained)
# -R^2 = 0 = model does not explain any variance
# -R^2 close to 1 = good model fit
# -R^2 close to 0 = poor model fit 

# In[17]:


#plot the scatter plot and overlay the fitted straight line using matplotlib 
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
#predicated response vector
y_hat = b0 + b1*x
#plotting the regression line
plt.plot(x, y_hat, color = "g")
plt.xlabel('x')
plt.xlabel('y')
plt.show()


# observations

# -The probability(p-value) for intercept (beta_0) is 0.707 > 0.05
# -Therefore the intercept coefficient may not be that much significant in prediction
# -However the p-value for "daily" (beta_1) is 0.00 < 0.05
# -theredore the beta_1 coefficent is highly significant and is contributint to prediction.

# In[18]:


model1.params


# In[19]:


print(f'model t-values:\n{model1.tvalues}\n--------------\nmodel p-values: \n{model1.pvalues}')


# In[20]:


(model1.rsquared,model1.rsquared_adj)


# Predict for new data Point

# In[21]:


newdata=pd.Series([200,300,1500])


# In[23]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[24]:


model1.predict(data_pred)


# In[31]:


pred = model1.predict(data1["daily"])
pred


# In[34]:


data1["Y_hat"] = pred
data1


# In[35]:


data1["residuals"]= data1["sunday"]-data1["Y_hat"]
data1


# In[36]:


mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy  as np


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# Descriptio of columns

# * MPG:Milege of the cars(Mile per gallon)(This is Y-column to be predicated)
# * HP: Horse power of the car(X1 column)
# * VOL: Volume of the car(size)(X2 column)
# * SP: Top speed of the car(Miles per Hour)(X3 column)
# * WT: Weight of the car(Pounds)(X4 column)

# In[3]:


cars = pd.DataFrame(cars,columns=["HP","VOL","SP","WT","MPG"])
cars.head()


#     EDA

# In[4]:


cars.info()


# In[5]:


#check for missing values
cars.isna().sum()


# In[6]:


#Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="HP", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="HP", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# #### Observation 
# - it is a right skewd plot
# - no.of outliers are 7

# In[7]:


# Create a figure with two subplots (one the above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})

# Creating a boxplot
sns.boxplot(data=cars, x="SP", ax=ax_box, orient='h')
ax_box.set(xlabel=' ') #Removes x label for the boxplot

#creating a histogram in th esame x-axis
sns.histplot(data=cars, x="SP", ax=ax_hist, bins=30,kde=True, stat="density")
ax_hist.set(ylabel='Density')

# Adjust layout'
plt.tight_layout()
plt.show()


# ### Observation
# - the outlier are the nature of the data it has outliers on both right and left side
# - no of outliers are 2

# # Create a figure with two subplots (one the above the other)
# fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios":(.15, .85)})
# 
# # Creating a boxplot
# sns.boxplot(data=cars, x="WT", ax=ax_box, orient='h')
# ax_box.set(xlabel=' ') #Removes x label for the boxplot
# 
# #creating a histogram in th esame x-axis
# sns.histplot(data=cars, x="WT", ax=ax_hist, bins=30,kde=True, stat="density")
# ax_hist.set(ylabel='Density')
# 
# # Adjust layout'
# plt.tight_layout()
# plt.show()

# In[8]:


cars[cars.duplicated()]


# In[9]:


# Pair plot
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[10]:


cars.corr(numeric_only=True)


# In[11]:


cars.corr()


# #observations from correlation plots and coeffcients

# * Between x and y ,all x variables are showing moderate to high correlation strengths,highest being between HP and MPG

# * Therefore this dataset qualifies for buliding a multiple linear regression model to predict MPG

# * Among x columns (x1,x2,x3,and x4) some very high correlation strengths are observed between SP vs HP, VOL vs WT

# * The high correlation among x columns x coumns is not desirable as it might lead to multicollinearity problem

# #preparing a preliminary model considering all x column

# In[12]:


import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[13]:


model1.summary()


# ####Observations from model summary

# * The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns

# * The probabaility value with respect to F-statistic is close to zero, indicating that all or some of X columns are significant

# * The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be futher explored

# #### performance metrics for model1

# In[14]:


# Find the performance metrics
# create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[16]:


#predict for the given X data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[19]:


#predict for the given X data columns
pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[20]:


#compute the Mean Squared Error (MSE), RMSE for model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# #### checking for multicollinearity among X-columns using VIF method

# In[21]:


#cars.head()


# In[22]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# #### Observations for VIF Values

# * The ideal range of VIF values shall be between 0 to 10 .However slightly higher values can be tolerated
# * As seen from the very high VIF values forVOL and WT, it is clear that they are prone to multicollinearity problem.
# * Hence it is decided to drop one of the columns (either VOL or WT) to overcome the multicollinearity.
# * It is decided to drop WT and retain VOL column in futher models

# In[23]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[25]:


model2 = smf.ols('MPG~HP+VOL+SP', data=cars1).fit()
model2.summary()


# #### Performance metrics for model2

# In[26]:


df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# pred_y2 = model2.predict(cars1.iloc[:,0:4])
# df2["pred_y2"] = pred_y2
# df2.head()

# In[28]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# #### Observations  from model2 summary()

# * The adjusted R-squared value improved slightly to 0.76
# * All the p-values for model parameters are less than 5% hence they are significant
# * Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# * There is no improvement in MSE value

# In[ ]:





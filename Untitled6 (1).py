#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy  as np


# In[3]:


cars = pd.read_csv("Cars.csv")
cars.head()


# Descriptio of columns

# * MPG:Milege of the cars(Mile per gallon)(This is Y-column to be predicated)
# * HP: Horse power of the car(X1 column)
# * VOL: Volume of the car(size)(X2 column)
# * SP: Top speed of the car(Miles per Hour)(X3 column)
# * WT: Weight of the car(Pounds)(X4 column)

# In[ ]:





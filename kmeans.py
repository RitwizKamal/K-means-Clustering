#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
df=pd.read_csv('C:/Users/Ritwiz/OneDrive/Desktop/iris.csv')
df.head()


# In[2]:


df.drop('species', axis=1, inplace = True)
df.drop('petal_width', axis=1, inplace = True)
df.head()


# In[3]:


#df.head()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#plt.style.use('ggplot')
new_array=df.values[:,0:3]
#colour=df.values[:,4]

ax.scatter(new_array[:,0],new_array[:,1],new_array[:,2])


# In[5]:


#colour


# In[6]:


import numpy as np

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(new_array)
labels=kmeans.predict(new_array)
ax1.scatter(new_array[:,0],new_array[:,1],new_array[:,2], c=labels.astype(np.float))


# In[7]:


labels


# In[ ]:





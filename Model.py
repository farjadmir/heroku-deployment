#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:


df = pd.read_csv('hiring.csv')
df.head()


# In[4]:


df['experience'].fillna(0, inplace = True)


# In[5]:


df


# In[6]:


df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(), inplace=True)


# In[7]:


df


# In[9]:


def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four': 4
                 ,'five':5, 'six':6, 'seven':7, 'eight': 8, 'nine':9, 'ten':10,'eleven':11, 0:0}
    return word_dict[word]

df['experience'] = df['experience'].apply(lambda x: convert_to_int(x))


# In[10]:


X = df.iloc[:,:3]
X


# In[11]:


y = df.iloc[:, -1]


# In[12]:


y


# In[13]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[14]:


regressor.fit(X,y)


# In[15]:


# saving moodel to Disk

pickle.dump(regressor, open('model.pkl', 'wb'))


# In[16]:


model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,8,6]]))


# In[ ]:





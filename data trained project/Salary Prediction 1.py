#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[117]:


df = pd.read_csv("D:\\project\\Salaries\\salaries.csv")
df


# In[118]:


df.head()


# In[119]:


df.tail()


# In[120]:


type(df)


# In[121]:


df.dtypes


# In[122]:


df.columns


# In[123]:


df.shape


# In[124]:


df.describe()


# In[125]:


df.isnull().sum()


# In[126]:


df.corr()


# In[127]:


corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


# In[128]:


df.mean()


# In[129]:


df['sex'].value_counts()


# In[130]:


df['sex'].value_counts(normalize=True)


# In[131]:


df['sex'].value_counts().plot.bar(title='Sex')


# In[132]:


df_copy=df.copy()
add = df_copy['rank']
df_copy.head()


# In[133]:


df_copy=df_copy.drop(['rank','yrs.since.phd'], axis=1)
df_copy.head()


# In[134]:


df_copy["discipline"] = df_copy["discipline"].replace('A','D') 
df_copy.head()


# In[135]:


df_copy['salary'].mean()


# In[136]:


df_copy['salary']=df_copy['salary'].fillna(df_copy['salary'].mean())
df_copy.head()


# In[137]:


df_sorted=df.sort_values(by='yrs.service')
df_sorted.head()


# In[138]:


df.sort_values(by = 'yrs.service', ascending = False, inplace = True)
df.head()


# In[139]:


df['discipline'] = df['discipline'].map({'A':1,'B':2})
df['discipline'].head(10)


# In[140]:


sns.distplot(df['salary']);


# In[141]:


df.groupby(['rank'])['salary'].count().plot(kind='bar')


# In[142]:


sns.set_style("whitegrid")

ax = sns.barplot(x='rank',y ='salary', data=df, estimator=len)


# In[143]:


ax = sns.barplot(x='rank',y ='salary', hue='sex', data=df, estimator=len)


# In[144]:


sns.regplot(x='yrs.service', y='salary', data=df)


# In[145]:


sns.pairplot(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





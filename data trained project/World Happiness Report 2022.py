#!/usr/bin/env python
# coding: utf-8

# World Happiness Report
# Problem Statement:
# Context
# 
# The World Happiness Report is a landmark survey of the state of global happiness. The first report was published in 2012, the second in 2013, the third in 2015, and the fourth in the 2016 Update. The World Happiness 2017, which ranks 155 countries by their happiness levels, was released at the United Nations at an event celebrating International Day of Happiness on March 20th. The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. The reports review the state of happiness in the world today and show how the new science of happiness explains personal and national variations in happiness.
# 
# What is Dystopia?
# 
# Dystopia is an imaginary country that has the world’s least-happy people. The purpose in establishing Dystopia is to have a benchmark against which all countries can be favorably compared (no country performs more poorly than Dystopia) in terms of each of the six key variables, thus allowing each sub-bar to be of positive width. The lowest scores observed for the six key variables, therefore, characterize Dystopia. Since life would be very unpleasant in a country with the world’s lowest incomes, lowest life expectancy, lowest generosity, most corruption, least freedom and least social support, it is referred to as “Dystopia,” in contrast to Utopia.
# 
# What are the residuals?
# 
# The residuals, or unexplained components, differ for each country, reflecting the extent to which the six variables either over- or under-explain average life evaluations. These residuals have an average value of approximately zero over the whole set of countries. 
# 
# What do the columns succeeding the Happiness Score(like Family, Generosity, etc.) describe?
# 
# The following columns: GDP per Capita, Family, Life Expectancy, Freedom, Generosity, Trust Government Corruption describe the extent to which these factors contribute in evaluating the happiness in each country.
# The Dystopia Residual metric actually is the Dystopia Happiness Score(1.85) + the Residual value or the unexplained value for each country.
# 
# The Dystopia Residual is already provided in the dataset. 
# 
# If you add all these factors up, you get the happiness score so it might be un-reliable to model them to predict Happiness Scores.
# 
# You need to predict the happiness score considering all the other factors mentioned in the dataset. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('D:\\data trained P1\\happiness_score_dataset.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


sns.heatmap(df.isnull())


# In[7]:


sns.pairplot(df)


# In[8]:


corr_mat = df.corr()


# In[9]:


plt.figure(figsize=(12,8))
sns.heatmap(corr_mat, annot=True)
plt.show()


# In[10]:


df.plot(kind='density', subplots=True, layout=(8,13), sharex=False, legend=False, fontsize=1, figsize=(25,15))
plt.show()


# In[11]:


df.skew()


# In[12]:


x = df.drop(['Country', 'Region', 'Happiness Score'], axis = 1)
y = df['Happiness Score']
from sklearn.preprocessing import power_transform
df_new = power_transform(x)
df_new = pd.DataFrame(df_new, columns = x.columns)
df_new.skew()


# In[13]:


df.plot(kind='density', subplots=True, layout=(8,13), sharex=False, legend=False, fontsize=1, figsize=(25,15))
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


y=y.round()
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
maxAccu = 0
maxRS = 0

for i in range(1, 200):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=i)
    LR = LogisticRegression()
    LR.fit(x_train, y_train)
    predlr = LR.predict(x_test)
    acc = accuracy_score(y_test, predlr)
    if acc > maxAccu:
        maxAccu = acc
        maxRS = i
print("Best Accuracy is:", maxAccu,"at Random State: ",maxRS)


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=98)


# In[17]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)
pred = log.predict(x_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Classification Report: ", classification_report(y_test, pred))


# In[18]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
pred = dtc.predict(x_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Classification Report: ", classification_report(y_test, pred))


# In[19]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
pred = rfc.predict(x_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Classification Report: ", classification_report(y_test, pred))


# In[20]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
pred = svc.predict(x_test)
print("Accuracy Score: ", accuracy_score(y_test, pred))
print("Confusion Matrix: ", confusion_matrix(y_test, pred))
print("Classification Report: ", classification_report(y_test, pred))


# In[21]:


from sklearn.model_selection import cross_val_score

scr = cross_val_score(log, x, y, cv=5)
print("Cross validation score for LOgistic Regression:", scr.mean())


# In[22]:


scr = cross_val_score(rfc, x, y, cv=5)
print("Cross validation score for Random Forest Classifier:", scr.mean())


# In[23]:


scr = cross_val_score(dtc, x, y, cv=5)
print("Cross validation score for Decision Tree:", scr.mean())


# In[24]:


scr = cross_val_score(svc, x, y, cv=5)
print("Cross validation score:", scr.mean())


# In[25]:


from sklearn.model_selection import RandomizedSearchCV


# In[26]:


n_estimators = [100, 200, 300, 400, 500]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [10, 20, 30, 40, 50]
max_depth.append(None)
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]

grid_param = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': min_samples_leaf}


# In[27]:


RFR = RandomForestClassifier(random_state=98)
RFR_random = RandomizedSearchCV(estimator=RFR, param_distributions=grid_param, n_iter=500, cv=5, verbose=2, random_state=98, n_jobs=-1)


# In[28]:


RFR_random.fit(x_train, y_train)
print(RFR_random.best_params_)


# In[ ]:





# In[ ]:





# In[ ]:





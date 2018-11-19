#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


salary_data=pd.read_csv("salary_data.csv")
x=salary_data.iloc[:,:-1].values
y=salary_data.iloc[:,1].values


# In[22]:


sns.barplot(x='YearsExperience',y='Salary',data=salary_data)


# In[55]:


salary_data


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[28]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[41]:


y_pred=lr.predict(x_test)
y_pred


# In[39]:


plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,lr.predict(x_train),color='red')
plt.title('Salary ~ Experience(Train Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[42]:


plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,y_pred,color='red')
plt.title('Salary ~ Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[45]:


from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))


# In[46]:


dataset=pd.read_csv('Social_Network_Ads.csv')


# In[47]:


dataset


# In[50]:


x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values


# In[51]:


y


# In[52]:


x


# In[57]:


sns.barplot(x='Age',y='EstimatedSalary',data=dataset)


# In[58]:


sns.heatmap(dataset.corr())


# In[61]:


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[62]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# In[ ]:





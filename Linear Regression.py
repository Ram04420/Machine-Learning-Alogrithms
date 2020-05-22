#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[35]:


df = pd.read_csv('Salary_Data.csv')


# In[36]:


df.head()


# In[40]:


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[43]:


from sklearn.linear_model import LinearRegression


# In[44]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# In[45]:


y_pred = reg.predict(X_test)


# In[50]:


plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Experince')
plt.ylabel('Salary')
plt.show()


# In[51]:


plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Experience vs Salary')
plt.xlabel('Experince')
plt.ylabel('Salary')
plt.show()


# In[ ]:





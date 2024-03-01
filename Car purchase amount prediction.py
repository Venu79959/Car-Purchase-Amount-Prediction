#!/usr/bin/env python
# coding: utf-8

# # Car purchasing amount prediction using ML by KODI VENU

# In[4]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
data=pd.read_csv('Car_Purchasing_Data.csv')


# Top 5 rows

# In[5]:


data.head()


# Last 5 rows

# In[6]:


data.tail()


# Dataset Shape

# In[7]:


data.shape


# In[8]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[9]:


data.info()


# Check null values in the dataset

# In[10]:


data.isnull().sum()


# Dataset Statistics

# In[11]:


data.describe()


# visualization

# In[12]:


import seaborn as sns
sns.pairplot(data)


# store feature matrix in X & response (target) in vector y

# In[13]:


data.columns


# In[15]:


X=data.drop(['Customer Name','Customer e-mail','Country','Car Purchase Amount'],axis=1)
y=data['Car Purchase Amount']


# Feature scaling

# In[16]:


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X_scaled=sc.fit_transform(X)
X_scaled
sc1=MinMaxScaler()
y_reshape=y.values.reshape(-1,1)
y_scaled=sc1.fit_transform(y_reshape)
y_scaled


# Train/Test split

# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.20,random_state=42)


# In[19]:


get_ipython().system('pip3 install xgboost')


# Import models

# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# Model Training

# In[22]:


lr=LinearRegression()
lr.fit(X_train,y_train)

svm=SVR()
svm.fit(X_train,y_train)

rf=RandomForestRegressor()
rf.fit(X_train,y_train)

gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)

xg=XGBRegressor()
xg.fit(X_train,y_train)


# Building ANN

# In[25]:


get_ipython().system('pip install tensorflow')


# In[30]:


import tensorflow.keras


# In[34]:


from keras.models import Sequential
from keras.layers import Dense


# Initialize ANN

# In[37]:


import warnings
warnings.filterwarnings('ignore')


# In[38]:


ann=Sequential()


# Adding input layer and first hidden layer

# In[39]:


ann.add(Dense(25,input_dim=5,activation='relu'))


# Adding second hidden layer

# In[41]:


ann.add(Dense(25,activation='relu'))


# Adding output layer

# In[42]:


ann.add(Dense(1,activation='linear'))


# Train ANN

# In[44]:


ann.summary()
ann.compile(optimizer='adam',loss='mean_squared_error')
ann.fit(X_train,y_train,epochs=100,batch_size=50,verbose=1,validation_split=0.2)


# Prediction

# In[45]:


y_pred1=lr.predict(X_test)
y_pred2=svm.predict(X_test)
y_pred3=rf.predict(X_test)
y_pred4=gr.predict(X_test)
y_pred5=xg.predict(X_test)
y_pred6=ann.predict(X_test)


# Evaluating the algorithm

# In[46]:


from sklearn import metrics
score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)
score5=metrics.r2_score(y_test,y_pred5)
score6=metrics.r2_score(y_test,y_pred6) 


# In[47]:


print(score1,score2,score3,score4,score5,score6)


# In[48]:


final_data=pd.DataFrame({'Models':['LR','SVR','RF','GR','XG','ANN'],'R2_SCORE':[score1,score2,score3,score4,score5,score6]})


# In[49]:


final_data


# In[51]:


import seaborn as sns
sns.barplot(x=final_data['Models'],y=final_data['R2_SCORE'])


# Save the model

# In[54]:


import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
ann=Sequential()
ann.add(Dense(25,input_dim=5,activation='relu'))
ann.add(Dense(25,activation='linear'))
ann.add(Dense(1,activation='linear'))
ann.compile(optimizer='adam',loss='mean_squared_error')
ann.fit(X_scaled,y_scaled,epochs=100,batch_size=50,verbose=1)


# In[65]:


import joblib
joblib.dump(ann,'car_model')
model=joblib.load('car_model')


# In[66]:


model


# Prediction on new data

# In[67]:


import numpy as np


# In[68]:


data.head(1)


# In[69]:


X_test1=sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))


# In[70]:


X_test1


# In[71]:


pred=ann.predict(X_test1)


# In[72]:


pred


# In[73]:


sc1.inverse_transform(pred)


# In[ ]:





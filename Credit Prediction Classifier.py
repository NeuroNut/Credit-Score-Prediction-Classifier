#!/usr/bin/env python
# coding: utf-8

# 
# ### Credit Score Prediction
# #### done by 
# #### Aditya Agarwal

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from graphviz import Source

from IPython.display import Image

import pandas_profiling


# In[2]:


test_data = pd.read_csv('test.csv')


# In[3]:


train_data = pd.read_csv('train.csv')


# In[4]:


profile = pandas_profiling.ProfileReport(train_data)
profile.to_notebook_iframe()
#analyzing raw dataset


# In[5]:


train_data.duplicated()


# In[6]:


train_data.info()


# In[7]:


train_data=train_data.fillna(0)


# In[8]:


train_data = train_data.convert_dtypes()
train_data.info()


# In[9]:


columns=['Annual_Income','Age','Monthly_Balance','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly']
train_data[columns]= train_data[columns].apply(pd.to_numeric, errors='coerce')


# In[10]:


train_data.info()


# In[11]:


train_data['Name']=train_data['Name'].astype('string')
train_data['Type_of_Loan']=train_data['Type_of_Loan'].astype('string')
train_data['Credit_History_Age']=train_data['Credit_History_Age'].astype('string')


# In[12]:


train_data.info()


# In[13]:


profile2 = pandas_profiling.ProfileReport(train_data)
profile2.to_notebook_iframe()
#analyzing dataset after training


# In[14]:


plt.figure(figsize = (20,15))
sns.heatmap(train_data.corr(),annot=True)


# In[66]:


plt.figure(figsize = (12,8))
plt.pie(train_data['Credit_Score'].value_counts(),labels=train_data['Credit_Score'].value_counts().index,autopct='%1.3f%%')
# plt.xlabel(size = 16)
plt.legend(loc = 'upper left',bbox_to_anchor=(1.5, 0., 0.5, 0.5))
plt.show()


# In[16]:


train_data['Name'].value_counts()


# In[17]:


train_data.groupby(['Name','Credit_Score']).count()['SSN']


# In[18]:


train_data.groupby(['Name','Type_of_Loan','Credit_Score']).count()['SSN']


# In[19]:


plt.figure(figsize = (8,6))
plt.scatter(train_data['Payment_of_Min_Amount'],train_data['Payment_Behaviour'])


# ## Seperating the result column and encoding rest of the data apply models

# In[20]:


x= train_data.drop('Credit_Score',axis=1)


# In[21]:


y= train_data['Credit_Score']
y.info()


# In[22]:


x


# In[23]:


x_string = x.select_dtypes(include=['string'])
x_string


# In[24]:


x_string=x_string.fillna('0')


# In[25]:


la = LabelEncoder()
for i in x_string.columns:
    x[i] = la.fit_transform(x[i])
    print(i)


# In[26]:


x=x.fillna(0)
x


# ## Training the classifing models and using HOLDOUT apporach to evaluate performance of models by dividing available data into training and testing/sample data

# In[27]:


x_train,x_sample,y_train,y_sample = train_test_split(x,y,test_size=.3)


# In[28]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[29]:


clf2 = RandomForestClassifier(n_estimators=100, random_state=0)
clf2.fit(x_train,y_train)


# ## visualizing the decision tree and importances in random forest

# In[30]:


plt.figure(figsize=(30,15))
plot_tree(clf, filled=True, rounded=True, class_names=np.unique(y), feature_names=x.columns)
plt.show()


# In[31]:


importances = clf2.feature_importances_
plt.barh(x.columns,importances)
plt.show()


# In[32]:


export_graphviz(clf, out_file="tree1.dot", feature_names=x.columns, class_names=y)


# In[33]:


with open("tree1.dot", "r") as f:
    dot_graph = f.read()
    
s = Source(dot_graph)
s.format = 'png'
s.render('tree1')


# In[34]:


Image(filename='tree1.png')


# ## evaluating and comparing the two trained models

# In[35]:


y_pre = clf.predict(x_sample)
y_pre


# In[36]:


y_pre2 = clf2.predict(x_sample)
y_pre2


# In[37]:


y_sample


# In[38]:


accuracy = np.mean(y_pre == y_sample)
print("Accuracy:", accuracy)
#accuracy of decision tree classification


# In[39]:


accuracy = np.mean(y_pre2 == y_sample)
print("Accuracy:", accuracy)
#accuracy of random forest method


# In[40]:


cm=confusion_matrix(y_sample,y_pre)
cm
#confusion matrix for decision tree


# In[41]:


cm2=confusion_matrix(y_sample,y_pre2)
cm2
#confusion matrix for random forest


# ## Deploying Random forest on test data

# In[42]:


test_data.info()


# In[44]:


profile3 = pandas_profiling.ProfileReport(test_data)
profile3.to_notebook_iframe()
#analyzing raw test dataset


# In[45]:


test_data=test_data.fillna(0)
test_data = test_data.convert_dtypes()
test_data.info()


# In[46]:


columns=['Annual_Income','Monthly_Balance','Age','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit','Outstanding_Debt','Amount_invested_monthly']
test_data[columns]= test_data[columns].apply(pd.to_numeric, errors='coerce')


# In[47]:


test_data.info()


# In[48]:


test_data['Name']=test_data['Name'].astype('string')
test_data['Type_of_Loan']=test_data['Type_of_Loan'].astype('string')
test_data['Credit_History_Age']=test_data['Credit_History_Age'].astype('string')


# In[49]:


t= test_data
t_string = t.select_dtypes(include=['string'])
for i in t_string.columns:
    t[i] = la.fit_transform(t[i])
    print(i)


# In[50]:


t=t.fillna(0)


# # Getting final result

# In[51]:


predicted_Score = clf.predict(t)


# In[52]:


predicted_Score


# In[53]:


ps=predicted_Score


# In[54]:


test_data['predicted_Score'] = ps


# In[67]:


plt.figure(figsize = (12,8))
plt.pie(test_data['predicted_Score'].value_counts(),labels=test_data['predicted_Score'].value_counts().index,autopct='%1.3f%%')

plt.legend(loc = 'upper left',bbox_to_anchor=(1.5, 0., 0.5, 0.5))
plt.show()


# In[56]:


test_data['Name'].value_counts()


# In[57]:


test_data.groupby(['Name','predicted_Score']).count()['SSN']


# In[58]:


test_data.groupby(['Name','Type_of_Loan','predicted_Score']).count()['SSN']


# In[59]:


plt.figure(figsize = (20,15))
sns.heatmap(test_data.corr(),annot=True)


# In[60]:


sns.violinplot(x = ps,y = 'Occupation',data = t)


# In[61]:


sns.violinplot(x = 'predicted_Score',y = 'Credit_Mix',data = test_data)


# In[62]:


sns.violinplot(x = 'predicted_Score',y = 'Name',data = test_data)


# In[ ]:





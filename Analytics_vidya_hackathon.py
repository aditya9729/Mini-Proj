
# coding: utf-8

# In[109]:


import numpy as np
import pandas as pd


# In[110]:


train=pd.read_csv('train.csv')
domain=train.Domain.str.split('.')
new_domain=[]
for i in domain:
    if len(i)==3:
        new_domain.append(i[1])
    elif len(i)==2:
        new_domain.append(i[0])
    elif len(i)==4:
        new_domain.append(i[1])
    elif len(i)==5:
        new_domain.append(i[1])
    else:
        new_domain.append(i[1])
        



# In[ ]:





# In[111]:


train.Domain=new_domain
train


# In[5]:


from sklearn.tree import DecisionTreeClassifier


# In[112]:


train.columns


# In[113]:


from sklearn.preprocessing import LabelEncoder
x=LabelEncoder()
train['newtag']=x.fit_transform(train.Tag)
t=LabelEncoder()
train['Domain']=t.fit_transform(train.Domain)


# In[114]:


X_train=train['Domain']
y_train=train['newtag']


# In[115]:


test=pd.read_csv('test.csv')


# In[116]:


domain=test.Domain.str.split('.')
new_domain=[]
for i in domain:
    if len(i)==3:
        new_domain.append(i[1])
    elif len(i)==2:
        new_domain.append(i[0])
    elif len(i)==4:
        new_domain.append(i[1])
    elif len(i)==5:
        new_domain.append(i[1])
    else:
        new_domain.append(i[1])
test.Domain=new_domain
test['Domain']=t.fit_transform(test.Domain)


# In[117]:


X_test=test.Domain
X_test=X_test[:,None]


# In[118]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
X_train=X_train[:,None]
model.fit(X_train,y_train)


# In[119]:


y=model.predict(X_test)


# In[120]:


predictions=x.inverse_transform(y)


# In[121]:


test.columns


# In[122]:


len(predictions)


# In[123]:


len(test['Webpage_id'])


# In[124]:


new_data=pd.read_csv('submission.csv')


# In[125]:


len(predictions)


# In[126]:


import csv
new_dir='submission3.csv'
csv=open(new_dir,'w')
columnTitleRow="Webpage_id, Tag\n"
csv.write(columnTitleRow)

for y in list(predictions):
    row=y+"\n"
    csv.write(row)


# In[127]:


pred=list(predictions)


# In[129]:


pred[25472:]



# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
# https://ndownloader.figshare.com/files/5975967 


# In[2]:


data=fetch_20newsgroups()
data.target_names


# In[3]:


categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
train=fetch_20newsgroups(subset='train',categories=categories)
test=fetch_20newsgroups(subset='test',categories=categories)
print(train.data[5])


# In[4]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(train.data,train.target)
labels=model.predict(test.data)
fig,ax=plt.subplots(figsize=(10,10))
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,fmt='d',cbar=False,annot=True,xticklabels=train.target_names,yticklabels=train.target_names)
plt.xlabel('Predictions');plt.ylabel('True values')
def predict_category(s,train=train,model=model):
    pred=model.predict([s])
    print(pred[0])
    return train.target_names[pred[0]]


# In[20]:


predict_category('sending a payload to the ISS')


# In[21]:


predict_category('Fraudulent transactions')


# In[22]:


predict_category('Hate Islam')


# In[5]:


predict_category('President Obama')


# In[6]:


predict_category('Microsoft')


# In[25]:


predict_category('Soccer')


# In[7]:


predict_category('Baseball')


# In[8]:


predict_category('sports')


# In[9]:


predict_category('microwave')


# In[10]:


predict_category('oven')


# In[11]:


predict_category('Pharmacy')


# In[12]:


predict_category('Bikes')


# In[13]:


from sklearn.metrics import classification_report


# In[14]:


print(classification_report(test.target,labels))


# In[16]:


predict_category('Narendra Modi')


# In[18]:


predict_category('Lebron James')


# In[19]:


predict_category('Ronaldo')


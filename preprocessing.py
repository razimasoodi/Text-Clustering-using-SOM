#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import re


# In[2]:


data=pd.read_csv('bbc-text.csv')
data=data.values


# In[3]:


text=[]
for i in data[ : ,-1]:
    text.append(i.split(' '))
stopwords= np.loadtxt(r'stopwords.txt',delimiter='\t',dtype=str)
stopwords=stopwords.reshape((1))
sw=stopwords[0].split(' ')
Doc2=[]
for doc in text:
    Doc1=[]
    for word in doc:
        old=''.join(re.findall("[a-zA-Z]+",word))
        Doc1.append(old)
    Doc2.append(Doc1)    
document=[]
e=np.zeros((data.shape[0],))
for i in range(len(Doc2)):
    tokens_without_sw=[word for word in Doc2[i] if word not in sw]   #stopword
    s=np.array([word for word in tokens_without_sw if len(word)>=3])  #shortword
    #print(tokens_without_sw[0])
    #s=s.astype(str)
    #t=np.unique(s)
    #print('t=',t)
    document.append(s)
    #e[i]=s


# In[4]:


word_list=[]
for doc in document:
    for word in doc:
        word_list.append(word)
w=list(np.unique(word_list))       


# In[5]:


tf=np.zeros((len(document),len(w)))
for i in range(len(document)):
    record=document[i]
    words,count=np.unique(record,return_counts=True)
    l=len(words)
    for j in range(l):
        word=words[j]
        andis=w.index(word)
        #print(andis)
        tf_value=count[j]/len(record)
        tf[i,andis] = tf_value             


# In[6]:


df=np.zeros((1,len(w)))
for i in range(len(w)) :
    for j in range(len(document)):
        if w[i] in document[j]:
            df[0,i]+=1
vij=np.log(1+tf)* np.log(len(document)/df)  


# In[7]:


np.save('som_data.npy', vij)


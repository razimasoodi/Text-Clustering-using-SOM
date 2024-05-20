#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import re
import random
import matplotlib.pyplot as plt


# In[3]:


def update_w(alpha,x,w):
    delta_w=alpha*(x-w)
    return delta_w


# In[4]:


def som(alpha,dataset,w):
    for t in range(10):
        indexes=list(np.arange(0,dataset.shape[0]))
        for i in range(dataset.shape[0]):
            indx=random.choice(indexes)
            data=dataset[indx]
            indexes.remove(indx)
            x=w@data
            winner_indx=np.argmax(x)
            #print(winner_indx)
            #min_list=[]
            #for j in range(w.shape[0]):
             #   min_list.append(np.linalg.norm(data-w[j]))
            #winner_indx=np.argmin(min_list)
            delta_w=update_w(alpha,data,w[winner_indx])
            w[winner_indx]=w[winner_indx]+delta_w  
        #alpha=0.99*alpha
    return w        


# In[5]:


def cluster(dataset,w,y): 
    d=dataset@w.T
    classes=[]
    classes1=[]
    for i in range(d.shape[0]):
        classes.append((np.argmax(d[i]),y[i]))
        classes1.append(np.argmax(d[i]))    

    classes1=np.array(classes1)   
    (unique_class, counts_class) = np.unique(classes1, return_counts=True)
    f1 = np.asarray((unique_class, counts_class)).T
    clusters=f1[ : ,1]
    return clusters,classes,f1


# In[6]:


def confution_matrix(classes):
    conf=np.zeros((5,5))
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            conf[i,j]=classes.count((i,j))
        m=np.argmax(conf[i])
        for k in range(conf.shape[1]):
            if i==k:
                t=conf[i,k]
                conf[i,k]=conf[i,m]
                conf[i,m]=t
    fig, ax = plt.subplots()
    im = ax.imshow(conf)
    #plt.imshow(clusters)
    for i in range(len(conf)):
        for j in range(len(conf)):
            text = ax.text(j, i, conf[i, j],ha="center", va="center", color="w",fontsize="xx-large")
    fig.tight_layout()
    plt.title("confusion-matrix")
   # plt.xlabel("label")
   # plt.ylabel("cluster")
    plt.show()  


# In[7]:


def hist(f1):
    courses = list(f1[ : ,0])
    values = list(f1[ : ,1])
    fig = plt.figure(figsize = (10, 5))
    plt.bar(courses, values, color ='maroon',width = 0.4)
    plt.title("Histogram")
    plt.xlabel("clusters")
    plt.ylabel("No. of samples")
    plt.show()


# In[8]:


def som_hits(cluster):
    clusters=cluster.reshape((-1,1))
    #print(clusters.shape)
    fig, ax = plt.subplots()
    im = ax.imshow(clusters)
    plt.imshow(clusters)
    for i in range(clusters.shape[0]):
        for j in range(clusters.shape[1]):
            text = ax.text(j, i , clusters[i,j] ,ha="center", va="center", color="w",fontsize="xx-large")
    fig.tight_layout()
    plt.title("som_hits")
   # plt.xlabel("label")
   # plt.ylabel("cluster")
    plt.show()  


# In[15]:


def confution_matrix2(classes):
    #labels=['tech','business','sport','entertainment','politics']
    conf=np.zeros((5,5))
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            conf[i,j]=classes.count((i,j))
    #df=pd.DataFrame(conf,[0,1,2,3,4],labels)        
    fig, ax = plt.subplots()
    im = ax.imshow(conf)
    #plt.imshow(clusters)
    for i in range(len(co)):
        for j in range(len(df)):
            text = ax.text(j, i, df.iloc[i, j],ha="center", va="center", color="w",fontsize="xx-large")
    fig.tight_layout()
    plt.title("confusion-matrix")
    plt.xlabel("label")
    plt.ylabel("cluster")
    plt.show()  


# In[14]:


dataset =np.load('som_data.npy')
alpha=0.1
#seed=1
w=np.random.rand(5*dataset.shape[1]).reshape((5,dataset.shape[1]))
data=pd.read_csv('bbc-text.csv')
#lb=np.array(data[ : ,0])
labels=['tech','business','sport','entertainment','politics']
for label in labels:
    data['category']=data['category'].replace([label],labels.index(label))
y=data.iloc[ : ,0].values
weight=som(alpha,dataset,w)
clusters,classes,f1=cluster(dataset,weight,y)
confution_matrix(classes)
hist(f1)
som_hits(clusters)
confution_matrix2(classes)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import re
import random
import matplotlib.pyplot as plt


# In[2]:


def update_w(alpha,x,w,win_indx,neighbour,var):
    win_indx=np.array(win_indx)
    neighbour=np.array(neighbour)
    djk=np.linalg.norm(win_indx-neighbour)
    hk=np.exp((-djk**2)/(2*(var**2)))
    delta_w=alpha*(hk*(x-w))
    return delta_w


# In[3]:


def som(alpha,dataset,w,var):
    epoch=10
    for t in range(epoch):
        indexes=list(np.arange(0,dataset.shape[0]))
        for i in range(dataset.shape[0]):
                indx=random.choice(indexes)
                data=dataset[indx]
                indexes.remove(indx)
                #x=w@data
                #winner_indx=np.argmax(x)
                min_list=[]
                index_list=[]
                for x in range(w.shape[0]):
                    for y in range(w.shape[0]):
                        #print(w[i,j])
                        min_list.append(data@w[x,y])
                        index_list.append((x,y))
                winner_indx=np.argmax(min_list)
                for x in range(w.shape[0]):
                    for y in range(w.shape[0]):
                        neighbour=[x,y]
                        win_indx=[index_list[winner_indx][0],index_list[winner_indx][1]]
                        #print('i=',i,'j=',j,update_w(alpha,data,w[index_list[winner_indx][0],index_list[winner_indx][1]],neighbour,var))
                        delta_w=update_w(alpha,data,w[win_indx[0],win_indx[1]],win_indx,neighbour,var)
                        w[x,y]=w[x,y]+delta_w  
        alpha=alpha*np.exp((-t/epoch))
        var=var*np.exp((-t/epoch))
    return w       


# In[4]:


def cluster(dataset,w,y): 
    classes=[]
    classes1=[]
    for i in range(dataset.shape[0]):
        q=w@dataset[i]
        z=np.argmax(q)
        classes.append(((z//w.shape[0],z%w.shape[0]),y[i]))
        classes1.append(z)
    classes1=np.array(classes1)   
    (unique_class, counts_class) = np.unique(classes1, return_counts=True)
    f1 = np.asarray((unique_class, counts_class)).T
    clusters=f1[ : ,1].reshape((w.shape[0],w.shape[0]))
    return clusters,classes


# In[5]:


def hits_plot(clusters):
    fig, ax = plt.subplots()
    im = ax.imshow(clusters)
    #plt.imshow(clusters)
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            text = ax.text(j, i, clusters[i, j],ha="center", va="center", color="w",fontsize="xx-large")
    fig.tight_layout()
    plt.title("som_hits")
    plt.show()        


# In[6]:


def confusion_matrix(classes,t):
    conf=np.zeros((t,t))
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            lb=[]
            for k in range(5):
                lb.append(classes.count(((i,j),k)))
        s=np.array(lb)
        m=np.argmax(s)
    #for j in range(conf1.shape[1]):
        conf[i,j]=m
    fig, ax = plt.subplots()
    im = ax.imshow(conf)
    #plt.imshow(clusters)
    for i in range(len(conf)):
        for j in range(len(conf)):
            text = ax.text(j, i, conf[i, j],ha="center", va="center", color="w",fontsize="xx-large")
    fig.tight_layout()
    plt.title("confusion-matrix")
    plt.show()  


# In[7]:


data=pd.read_csv('bbc-text.csv')
labels=['tech','business','sport','entertainment','politics']
for label in labels:
    data['category']=data['category'].replace([label],labels.index(label))
y=data.iloc[ : ,0].values
dataset=np.load('som_data.npy')
var=1
#((3*3/2)+1)/(3*3)
alpha=0.01
neurons=[3,4,5]
dis=[]
for i in neurons:
    #seed=1
    w=np.random.uniform(-1,1,i*i*dataset.shape[1]).reshape((i,i,dataset.shape[1]))
    weights=som(alpha,dataset,w,var)
    clusters,classes=cluster(dataset,weights,y)
    distance_list=[]
    for j in range(len(classes)):
        distance_list.append(np.linalg.norm(dataset[j]-weights[classes[j][0]][classes[j][1]]))
    dis.append(sum(distance_list)/len(dataset)) 
    hits_plot(clusters)
    confusion_matrix(classes,i)


# In[8]:


print('Euclidean distance of all documents with 3*3 neurons: ',dis[0])
print('Euclidean distance of all documents with 4*4 neurons: ',dis[1])
print('Euclidean distance of all documents with 5*5 neurons: ',dis[2])


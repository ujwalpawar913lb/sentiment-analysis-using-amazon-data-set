# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:56:23 2020

@author: Ujwal Pawar
"""

# In[1]:


import numpy as np
import pandas as pd
import string
import re


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#from gensim.models import KeyedVectors

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# In[2]:


df=pd.read_csv("amazon_cells_labelled.txt",sep="\t",names=["text","target"])

# In[4]:


PUNCT_TO_REMOVE = string.punctuation
def func_remove_punct_num2(string):
    
    arr=[]
    for i in range(len(PUNCT_TO_REMOVE)):
        arr.append(PUNCT_TO_REMOVE[i])
#     print(arr)
    punct="\\"+"|\\".join(arr) # Creating exact regex
#     print(punct)

    pattern2=re.compile(punct)
    
    string=pattern2.sub(" ",string)
    string=re.sub('[0-9]'," ",string)
    return(string)


# In[5]:


def pre_process(string):
    string=str(string).lower()
    string=func_remove_punct_num2(string)
#     string=remove_stopwords(string)
    return string

def pre_processing(text_col,target=None):
    return text_col.apply(lambda x:pre_process(x))


# In[6]:


cust_transform=FunctionTransformer(pre_processing)


# In[7]:


model_list=[LogisticRegression(fit_intercept=True),RandomForestClassifier(),SVC()]
pipelines=[]
for model in model_list:
    pipe=Pipeline(steps=[('preprocessor',cust_transform),
                        ('tfidf',TfidfVectorizer(smooth_idf=True,use_idf=True)),
                        ('model',model)])
    pipelines.append(pipe)


# # word2vec pipeline

# In[14]:


#filename = 'GoogleNews-vectors-negative300.bin' # need to have this file locally
#word2vec = KeyedVectors.load_word2vec_format(filename, binary=True,limit=1000000)


# In[15]:

'''
def word2vec_func_mean(text_col,target=None):
    return np.array([
            np.mean([word2vec[w] for w in words if w in word2vec] or [np.zeros(300)], axis=0)
            for words in text_col.apply(lambda x: x.split())
                    ])


# In[16]:


word2vec_transform_mean=FunctionTransformer(word2vec_func_mean)


# In[17]:


model_list=[LogisticRegression(fit_intercept=True),RandomForestClassifier(),SVC()]
pipelines_w2v=[]

for model in model_list:
    pipe_w2v=Pipeline(steps=[('preprocessor',cust_transform),
                            ('w2v_transformer_mean',word2vec_transform_mean),
                            ('model',model)])
    pipelines_w2v.append(pipe_w2v)

'''
# In[26]:


import pickle


# In[30]:


final_model=pipelines[2].fit(df['text'],df['target'])


# In[36]:


pickle.dump(final_model,open('sentiment_analysis_model.pkl','wb'))








#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 数据加载
train_data = pd.read_csv('./train_data_public.csv')
#train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x: x.split(' '))
#train_data['train_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)
train_data['text']


# In[2]:


#train_data.loc[0, 'text']
train_data['text'] = train_data['text'].str.replace('\n', '')
train_data = train_data[['class', 'text']]
train_data.to_csv('./ernie_train.tsv', index=False, sep='\t')


# In[3]:


test_data = pd.read_csv('./test_public.csv')
#test_data['test_data'] = test_data.apply(lambda row: (list(row['text'])), axis=1)
test_data
test_data['text'] = test_data['text'].str.replace('\n', '')
test_data['class'] = 0
test_data = test_data[['class', 'text']]
test_data.to_csv('./ernie_test.tsv', index=False, sep='\t')


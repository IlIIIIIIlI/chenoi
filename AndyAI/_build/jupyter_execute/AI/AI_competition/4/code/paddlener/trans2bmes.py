#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 数据加载
train_data = pd.read_csv('./train_data_public.csv')
train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x: x.split(' '))
train_data['train_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)
train_data


# In[2]:


def tran2bmes(train_data, filename):
    file = open(filename, 'w', encoding='utf-8')
    for index, row in train_data.iterrows():
        for char, bio in zip(row['train_data'][0], row['BIO_anno']):
            # 空格
            char = char.strip()
            if char !='' and char!=' ' and char!=' ':
                file.write('{} {}\n'.format(char, bio))
        file.write('\n')
    file.close()
# 生成训练集
#tran2bmes(train_data, 'train.bmes')


# In[3]:


test_data = pd.read_csv('./test_public.csv')
test_data['test_data'] = test_data.apply(lambda row: (list(row['text'])), axis=1)
test_data


# In[4]:


filename = './test.bmes'
file = open(filename, 'w', encoding='utf-8')
for index, row in test_data.iterrows():
    for char in row['test_data']:
        # 空格
        char = char.strip()
        if char !='' and char!=' ' and char!=' ':
            file.write('{} {}\n'.format(char, 'O'))
    file.write('\n')
file.close()


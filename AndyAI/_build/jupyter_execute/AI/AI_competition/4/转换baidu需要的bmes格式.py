#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 数据加载
train = pd.read_csv('./train_data_public.csv')
train['BIO_anno'] = train['BIO_anno'].apply(lambda x : x.split(' '))
# 按照bio顺序加入词
train['train_Data'] = train.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)


# In[2]:


train


# In[3]:


def tran2bems(train, filename):
    file = open(filename, 'w', encoding='utf-8')
    for index, row in train.iterrows():
        for char, bio in zip(row['train_Data'][0], row['BIO_anno']):
            # 空格会干扰正常的读取，
            char = char.strip()
            if char != '' and char !=' ' and char!=' ':
                file.write('{} {}\n'.format(char, bio))
        file.write('\n')
    file.close()


# In[4]:


# 训练集生成文件
tran2bems(train, "train.bmes")


# In[5]:


test = pd.read_csv('./test_public.csv')


# In[6]:


# 测试集，标点符号要的，是o
test['test_Data'] = test.apply(lambda row: (list(row['text'])), axis=1)


# In[7]:


filename = './test.bmes'
file = open(filename, 'w', encoding='utf-8')
for index, row in test.iterrows():
    for char in row['test_Data']:
        # 空格
        char = char.strip()
        if char!='' and char!=' ' and char!=' ':
            file.write('{} {}\n'.format(char, 'O'))
    file.write('\n')
file.close()


# In[ ]:





# In[ ]:





# In[ ]:





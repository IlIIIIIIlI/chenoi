#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv('./ner_work/train_data_public.csv')
test = pd.read_csv('./ner_work/test_public.csv')
train


# In[2]:


get_ipython().system('pip install paddlehub==2.1.1')


# In[3]:


import paddle
import paddlehub as hub
from paddlehub.datasets.base_nlp_dataset import BaseNLPDataset
from typing import Dict, List, Optional, Union, Tuple
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset
from paddlehub.text.bert_tokenizer import BertTokenizer
from paddlehub.text.tokenizer import CustomTokenizer

# 自定义数据集
class MyDataset(TextClassificationDataset):
    """DemoDataset"""
    def __init__(self, tokenizer: Union[BertTokenizer, CustomTokenizer], max_seq_len: int = 128, mode: str = 'train'):
        # 数据集存放位置
        base_path = "/home/aistudio/ner_work"
        if mode == 'train':
            data_file = 'ernie_train.tsv'
        elif mode == 'test':
            data_file = 'ernie_test.tsv'
        else:
            data_file = 'ernie_dev.tsv'
        super().__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=["0", "1", "2"],
            is_file_with_header=True)

#Step2，使用PaddleHub
# ernie1.0, 2.0, 3.0
# PaddleHub 预训练包，直接用 + FineTunning, 多种预训练模型 sequence classification => sentence classification
model = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=3)
#数据集设置
train_dataset = MyDataset(tokenizer=model.get_tokenizer(), max_seq_len=512, mode='train')
#test_dataset = MyDataset(tokenizer=model.get_tokenizer(), max_seq_len=512, mode='test')
#val_dataset
#Step3，设置优化器，运行配置
optimizer = paddle.optimizer.AdamW(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='./ernie_checkpoint', use_gpu=True)


# In[ ]:


help(trainer)


# In[12]:


# 1,自定义数据集 2，sampling dataloader 3, 参数配置 4, 训练过程
trainer.train(
    train_dataset, # 只针对train
    epochs=3,
    batch_size=32,
    save_interval=1,
)


# In[4]:


label_map = {0: 0, 1: 1, 2: 2}
model = hub.Module(
    name='ernie_tiny',
    version='2.0.1',
    task='seq-cls',
    load_checkpoint='./ernie_checkpoint/epoch_3/model.pdparams', 
    label_map=label_map)
model


# In[5]:


train['text']


# In[ ]:


#test = pd.read_csv('./ner_work/test_public.csv')
train_results = model.predict([[x] for x in train['text'].values], max_seq_len=512, batch_size=32, use_gpu=True)


# In[ ]:


train_results


# In[6]:


test_result = pd.read_csv('./baselin1.csv')
#test_result['class'] = 

test = pd.read_csv('./ner_work/test_public.csv')
results = model.predict([[x] for x in test['text'].values], max_seq_len=512, batch_size=32, use_gpu=True)


# In[11]:


#results
test_result['class'] = results
test_result


# In[12]:


test_result.to_csv('./baseline2.csv', index=False)


# In[15]:


df = pd.read_csv('ner_work/train_data_public.csv')
df['class'].value_counts()
# 对于样本不均衡， kappa 敏感性指标
# label=2 造成偏袒 => Loss weight降低
# weight
# loss
# BERT Classification 纯情感分类score = 0.41 (没对loss优化)
# BERT Classification 纯情感分类score > 0.41 (loss优化)


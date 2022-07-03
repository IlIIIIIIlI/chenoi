#!/usr/bin/env python
# coding: utf-8

# # 产品观点提取

# PDF请访问[产品观点提取1.pdf](https://www.dropbox.com/s/mfovj0uw7obygad/L8%20%E4%BA%A7%E5%93%81%E8%AF%84%E8%AE%BA%E8%A7%82%E7%82%B9%E6%8F%90%E5%8F%961V0.2.pptx?dl=0)。[产品观点提取2.pdf](https://www.dropbox.com/s/vhmyi86zaicj45m/L9%20%E4%BA%A7%E5%93%81%E8%AF%84%E8%AE%BA%E8%A7%82%E7%82%B9%E6%8F%90%E5%8F%962V0.2.pptx?dl=0)。

#  对于哪一款产品的什么态度

# 产品
# 评价名词
# 评价形容词
# 银行

# 标注，名词问题，不统一问题，很主观
# 
# 
# 2个任务，bio实体标注（begin in & out） 和 情感分类
# 
# 
# 总分排名

# precision 和 recall 都有作弊的可能
# 
# 
# F1 结合两者
# 
# 
# Kappa系数 - 情感 - 越小，越不敏感 - 考虑了不平衡的样本
# 
# 
# accuracy在样本不均衡的时候没卵用

#  NER 命名实体，从文本中找到你想要的任何东西，sequence tagging

# 信息提取，问答系统，句法分析，机器翻译等多种ML的重要基础
# 
# 深度学习
# 
# LSTM / BI-LSTM
# 
# 
# LSTM - CRF
# 
# 
# BERT - CRF

# 在外面

#  维特比的方法来融合两个模型
# 
# CRF 计算transition函数

# # pytorch 里面定义了一种bilstm-crf的方法

# In[2]:


# read data
import pandas as pd
train = pd.read_csv('./train_data_public.csv')
test = pd.read_csv('./test_public.csv')


# In[3]:


# 给BIO数据切分一下
train['BIO_anno'] = train['BIO_anno'].apply(lambda x : x.split(' '))


# In[4]:


# 按照bio顺序加入词
train['train_Data'] = train.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)


# In[5]:


train


# In[6]:


# 测试集，标点符号要的，是o
test['test_Data'] = test.apply(lambda row: (list(row['text'])), axis=1)


# In[7]:


train_txt = []
for i in range(len(train)):
    train_txt.append(train.loc[i, 'train_Data'])

test_txt = []
for i in range(len(test)):
    test_txt.append(test.loc[i, 'test_Data'])


#  加载pytorch

# In[8]:


# !pip install torch


# # 查看pytorch版本

# In[9]:


import torch
torch.__version__


# In[10]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# 返回向量中最大值索引
def argmax(vec):
    # return argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# 将句子转换成IDlist
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# 计算log sum exp
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # 将bilstm提取的特征向量vec映射到特征空间，得到了发射分数
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 从j 到 i的转移分数得分
        # 转移矩阵的参数初始化
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 初始化到start的分数非常小，因此不可能转移过来
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # stop点也可能转移到其他tag
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    # 初始化参数
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # 通过前向传播进行计算
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        # 初始化位置0的发射分数
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        # 迭代整个句子
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 当前路径分数 = 之前时间步 + 转移分数 + 发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                # 对当前分数计算 loss function （log_sum_exp)
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 更新forward_var
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到stop
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 计算最终分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # 通过bi_LSTM提取特征
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 计算tag序列的分数，一条路径的分数
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 不断递推计算
            score = score +                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # 求解最优路径
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # 只考虑上一步和上一步的转移
                # 维特比记录最优路径，考虑上一步的分数以及上一步tag转移到当前tag的分数
                # 不用考虑当前的分数
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # 损失函数的组成
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    # 通过bilstm计算发射分数
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


#  Running Training

# In[11]:


# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# # 隐藏层的神经元
# EMBEDDING_DIM = 11
# HIDDEN_DIM = 6

# # Make up some training data
# training_data = train_txt[:10000] #使用全量的

# word_to_ix = {}
# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

# len(word_to_ix)


# In[12]:


# # testing data --> 将汉字转换成id
# testing_data = test_txt[:10000] #使用全量的

# for sentence in testing_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

# len(word_to_ix)


# In[13]:


# import pickle
# with open('./word_to_id.pkl', 'wb') as file:
#     pickle.dump(word_to_ix, file)


# 这个bio的数据标注
# 一句话，里面不同词性填到不同的列
# 产品，名词，形容词，银行
# 算法转换这些词性为不同的列

# In[14]:


# tag_to_ix = {"B-BANK": 0, "I-BANK": 1, "B-PRODUCT": 2, "I-PRODUCT": 3, "O": 4, "B-COMMENTS_N":5, "I-COMMENTS_N":6, "B-COMMENTS_ADJ":7, "I-COMMENTS_ADJ":8, START_TAG:9, STOP_TAG:10}

# '''
# B-BANK 代表银行实体的开始
# I-BANK 代表银行实体的内部
# B-PRODUCT 代表产品实体的开始
# I-PRODUCT 代表产品实体的内部
# O 代表不属于标注的范围
# B-COMMENTS_N 代表用户评论（名词）
# I-COMMENTS_N 代表用户评论（名词）实体的内部
# B-COMMENTS_ADJ 代表用户评论（形容词）
# I-COMMENTS_ADJ 代表用户评论（形容词）实体的内部
# '''

# model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# # 用随机参数进行预测, 不准确，只是跑了一遍流程
# # Check predictions before training
# with torch.no_grad():
#     # 句子汉字 --》 ID LIST
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     # 使用model预测BIO类别
#     print(model(precheck_sent))


#  减少误差的大小

# In[15]:


# from tqdm import tqdm
# # Make sure prepare_sequence from earlier in the LSTM section is loaded
# # 40多轮的结果得出的结果会比较好一点
# for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in tqdm(training_data):
#         # Step 1. Remember that Pytorch accumulates gradients.
#         # We need to clear them out before each instance
#         # 梯度清零，防止梯度爆炸
#         model.zero_grad()

#         # Step 2. Get our inputs ready for the network, that is,
#         # turn them into Tensors of word indices.
#         # 原始文字 =》 IDX
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

#         # Step 3. Run our forward pass.
#         loss = model.neg_log_likelihood(sentence_in, targets)

#         # Step 4. Compute the loss, gradients, and update the parameters by
#         # calling optimizer.step()
#         # 因为之前是前向转播，这里我们反向传播更新参数
#         loss.backward()
#         optimizer.step()
#     # 我们需要保存一下我们的运行结果, 以下代码意思是运行多少轮保存一次
#     if (epoch+1)%1==0:
#         file_name='model{}.pt'.format(epoch+1)
#         torch.save(model, file_name)
#         prepare_sequencerint('{ saved}'.format(file_name))


# In[16]:


# # 也可以.cuda放到GPU里面、 也可以放到paddle里面
# # Check predictions after training
# # 得出的结果更科学一点
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     print(model(precheck_sent))
# # We got it!


#  放到GPU 用to(device), 模型和数据都要

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 数据加载
train_data = pd.read_csv('./train_data_public.csv')
train_data


# In[2]:


len(train_data.loc[1, 'BIO_anno']) # 19


# In[3]:


train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x: x.split(' '))
train_data['train_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)
train_data


# In[4]:


test_data = pd.read_csv('./test_public.csv')
test_data


# In[5]:


test_data['test_data'] = test_data.apply(lambda row: (list(row['text'])), axis=1)
test_data


# In[6]:


train_data_txt = []
test_data_txt = []
for i in range(len(train_data)):
    train_data_txt.append(train_data.loc[i, 'train_data'])
    
for i in range(len(test_data)):
    test_data_txt.append(test_data.loc[i, 'test_data'])
print(train_data_txt[0])
print(test_data_txt[0])


# In[7]:


import torch
torch.__version__


# In[8]:


import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# 返回vec中的最大值索引
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# 将句子转换为ID list的形式
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# log_sum_exp损失函数
# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# 创建模型
class BiLSTM_CRF(nn.Module):
    # 模型初始化
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
        
        # 将BiLSTM提取的特征向量vec 映射到特征空间，即得到了 发射分数
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移矩阵的参数初始化, transitions[i, j]代表是从第j个tag 转移到第i个tag的转移分数
        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # 初始化所有tag 到START_TAG的分数= -10000（非常小），即不可能由其他tag转移到START_TAG
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        # 初始化STOP_TAG转移到其他tag的分数=-10000（非常小），即不可能由STOP_TAG转移到其他TAG
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 初始化BiLSTM的参数
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # 通过前向传播进行计算
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # 初始化 STEP0，即START_TAG位置的 发射分数
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # 迭代整个句子
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # 取出当前tag的发射分数，和之前时间步的tag没有关系
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                
                # 当前路径的分数 = 之前时间步的分数 + 转移分数 + 发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                # 对当前分数计算 loss function(log_sum_exp)
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
                
            # 更新forward_var
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 考虑最终转移到STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # 计算最终分数
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # 通过Bi-LSTM提取特征
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # 计算给定tag序列的分数，也就是一条路径的分数
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            # 递推计算路径分数： 转移分数 + 发射分数
            score = score +                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    # 使用viterbi算法，求解最优路径（也就是累计分数最大的）
    def _viterbi_decode(self, feats):
        backpointers = []

        # 初始化viterbi变量
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            # 保存当前时间步的回溯指针
            bptrs_t = []  # holds the backpointers for this step
            # 保存当前时间步的viterbi变量
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # viterbi算法 记录最优路径，只考虑上一步的分数，以及上一步tag转移到当前tag的转移分数
                # 不用考虑当前tag的发射分数（共用的，求解最大值的使用不用考虑）
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
            # 更新了forward_var，加上当前tag的发射分数
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # 考虑转移到STOP_TAG的转移分数
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # 通过回溯指针，找到最优路径
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            # 最优路径，存储到best_path
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    # 损失函数
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 通过BiLSTM计算发射分数
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        # 通过viterbi算法找到最优路径
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[9]:


#train_data_txt[:100]
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 11
HIDDEN_DIM = 6

# Make up some training data
training_data = train_data_txt[:10000] # 使用全量的

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

len(word_to_ix)


# In[10]:


# 将汉字 => ID
testing_data = test_data_txt[:10000] # 使用全量的

#word_to_ix = {}
for sentence in testing_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
len(word_to_ix)


# In[11]:


import pickle
with open('./word_to_ix.pkl', 'wb') as file:
    pickle.dump(word_to_ix, file)


# In[13]:


# tag字典
tag_to_ix = {"B-BANK": 0, "I-BANK": 1, "B-PRODUCT": 2, "I-PRODUCT": 3, "O": 4, "B-COMMENTS_N": 5,
             "I-COMMENTS_N": 6, "B-COMMENTS_ADJ": 7, "I-COMMENTS_ADJ": 8,
             START_TAG: 9, STOP_TAG: 10}

# 创建BiLSTM_CRF模型
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# 在训练之前，用随机的参数进行预测
# Check predictions before training
with torch.no_grad():
    # 将句子的汉字 => ID list
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(training_data[0][0])
    print('precheck_sent=', precheck_sent)
    # precheck_tags：BIO类别
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    # 使用model预测BIO类别
    print(model(precheck_sent))


# In[14]:


from tqdm import tqdm
# NER， BIO标注
#print(training_data[0][0])
#word_to_ix
#precheck_sent
#precheck_tags

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in tqdm(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        # 梯度清零，防止梯度爆炸
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        # 原始汉字 => idx
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
    # 模型保存
    if (epoch+1)%1==0:
        file_name = 'model{}.pt'.format(epoch+1)
        torch.save(model, file_name)
        print('{} saved'.format(file_name))


# In[ ]:


with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


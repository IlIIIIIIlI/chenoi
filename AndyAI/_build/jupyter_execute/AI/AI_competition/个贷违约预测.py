#!/usr/bin/env python
# coding: utf-8

# # 个贷违约

# PDF请访问[个贷违约预测.pdf](https://www.dropbox.com/s/ms0sefr2sulvmdg/L3%20%E4%B8%AA%E8%B4%B7%E8%BF%9D%E7%BA%A6%E9%A2%84%E6%B5%8B1V0.2.pptx?dl=0)。

# ## 使用autoML模型

# In[ ]:


# # 用autoML 来找最佳的metric - path 只是model信息保存的文件路径
# model = TabularPredictor(label="subscribe", eval_metric='f1', path="model_simple")


# In[ ]:


# # 获取模型各个feature的重要性
# model.feature_metadata_in.get_features()
# model.feature_importance(train)


#  ## 特征工程：考虑每个feature实际的重要性

# ### 一种numeric格式转换成categoric的格式的方式
# for a list of pred, makes it be a human-readable labels and input to a dataframe as a column
# 
# y_pred = [1 if x >= 0.5 else 0 for x in y_pred]
# 
# subscribe_map = {1.0:"yes", 0.0:"no"}
# 
# result['subscribe'] = [subscribe_map[x] for x in y_pred]

# In[ ]:


# 最后一位按照年加进去
def f(x):
    try:
        x1, x2, x3 = x.split('-')
        if x1 == '2022':
            x = str(int(x3) + 2000) + '/' + x2  + '/' + '1'
    except Exception:
        x == x
    return x
train_public['earlies_credit_mon'] = train_public['earlies_credit_mon'].apply(f)
test_public['earlies_credit_mon'] = test_public['earlies_credit_mon'].apply(f)


# In[1]:


def process_cat(data):
    # 	class	employer_type	industry	work_year
    # 先看class有多少类
    class_map = {'A':6, 'B':5, 'C':4, 'D':3, 'E':2, 'F':1, 'G':0}
    yr_map = {'< 1 year':0, '1 years':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
    industry_map = {'金融业':0, '房地产业':0, '电力、热力生产供应业':1, '制造业':1, '建筑业':1, '交通运输、仓储和邮政业':1, '农、林、牧、渔业':2, '采矿业':2, '批发和零售业':3, '住宿和餐饮业':3, '文化和体育业':3, '公共服务、社会组织':4, '国际组织':4, '信息传输、软件和信息技术服务业':5}
    emp_map = {'普通企业':0, '政府机构':1, '世界五百强':3, '高等教育机构':1, '上市企业':2, '幼教与中小学校':1}
    '''
    单纯的用于表示cat的numeric变成程度关联
        金融业                122758
        电力、热力生产供应业   92072
        公共服务、社会组织     77326
        住宿和餐饮业           68422
        文化和体育业           61548
        信息传输、软件和信息技术服务业     61023
        建筑业                53330
        房地产业              45733
        交通运输、仓储和邮政业 38300
        采矿业                38092
        农、林、牧、渔业       38091
        国际组织              22909
        批发和零售业          22738
        制造业                22658
        普通企业             347404
        政府机构             197443
        幼教与中小学校        76547
        上市企业             76425
        世界五百强           41361
        高等教育机构         25820
    '''
    # 其他几个类用labelencoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
#     data['employer_type'] = le.fit_transform(data['employer_type'])
    data['industry'] = le.fit_transform(data['industry'])
#     data['industry'] = data['industry'].map(industry_map)
    data['class'] = data['class'].map(class_map)
    data['work_year'] = data['work_year'].map(yr_map)
    data['employer_type'] = data['employer_type'].map(emp_map)
    
    # 看到有时间，就先处理时间类型
    import datetime
    data['earlies_credit_mon'] = pd.to_datetime(data['earlies_credit_mon'])
    # 时间多尺度, 都是月度时间，用value_counts()可以看到日期日的话都是1
    data['earlies_credit_mon_yr'] = data['earlies_credit_mon'].dt.year
    data['earlies_credit_mon_mt'] = data['earlies_credit_mon'].dt.month

    # 将时间转换成时间区间
    # 首先设置basetime
    basetime1 = data['earlies_credit_mon'].min()
    data['earlies_credit_mon_diffs'] = data['earlies_credit_mon'].apply(lambda x:x-basetime1).dt.days
    # train['issue_date_diffs']

    data['issue_date'] = pd.to_datetime(data['issue_date'])
    # 时间多尺度, 都是月度时间，用value_counts()可以看到日期日的话都是1
    data['issue_year'] = data['issue_date'].dt.year
    data['issue_month'] = data['issue_date'].dt.month

    # 将时间转换成时间区间
    # 首先设置basetime
    basetime2 = data['issue_date'].min()
    data['issue_date_diffs'] = data['issue_date'].apply(lambda x:x-basetime2).dt.days
    # train['issue_date_diffs']
    
    # drop多余的列
    # data.drop(['issue_date', 'earlies_credit_mon'], axis=1, inplace=True)
    
    return data


# In[5]:


# 如果打印出来这两个table， 发现有些feature是一样的但是名字不一样
train_public = train_public.rename(columns={'isDefault':'is_default'})


# In[6]:


# 查找数据集ab 
common_cols = []
for col in train_public.columns:
    if col in train_internet:
        common_cols.append(col)
len(common_cols)


# In[7]:


print(set(train_internet.columns) - set(common_cols))


# In[8]:


import numpy as np

train_internet = train_internet.drop(['work_type', 'marriage', 'sub_class', 'f5', 'house_loan_status', 'offsprings'], axis=1)
for col in ['known_outstanding_loan', 'app_type', 'known_dero']:
    train_internet[col] = np.nan

# 增加 is_internet 字段
train_internet['is_internet'] = True
train_public['is_internet'] = False
test_public['is_internet'] = False

# 拼接所有数据
df = pd.concat([train_public, test_public, train_internet]).reset_index(drop=True)


# In[9]:


df = process_cat(df)

# ---------- # 
'''['loan_id', 'user_id', 'total_loan', 'year_of_loan', 'interest',
       'monthly_payment', 'class', 'employer_type', 'industry', 'work_year',
       'house_exist', 'censor_status', 'issue_date', 'use', 'post_code',
       'region', 'debt_loan_ratio', 'del_in_18month', 'scoring_low',
       'scoring_high', 'pub_dero_bankrup', 'early_return',
       'early_return_amount', 'early_return_amount_3mon', 'recircle_b',
       'recircle_u', 'initial_list_status', 'earlies_credit_mon', 'title',
       'policy_code', 'f0', 'f1', 'f2', 'f3', 'f4', 'is_default',
       'known_outstanding_loan', 'app_type', 'known_dero', 'is_internet'],
'''

df['monthly_loan_rb'] = df['monthly_payment'] * 12 * df['year_of_loan'] / (df['recircle_b'] + 0.1)
# 单位循环额度利用率
df['rurb'] = df['recircle_u'] / (df['recircle_b'] + 0.1)
# debt_loan_ratio 债务收入比
# 对于 <0 的异常值进行处理 ==》 转换为正数
df['debt_loan_ratio'] = np.abs(df['debt_loan_ratio']).fillna(df['debt_loan_ratio'].mean())
# recircle_b_total_loan 信贷额度周转率
df['recircle_b_total_loan'] = df['recircle_b'] / (df['total_loan'] + 0.1)
# recircle_b_monthly_pmt 分期付款金额
df['recircle_bmp'] = df['recircle_b'] / (df['monthly_payment'] + 0.1)
# 空缺情况, =1 统计有多少空列 =0 统计有多少空行
df['nulls'] = df.isnull().sum(axis=1)
# 平均每年贷款金额
df['avg_loan'] = df['total_loan'] / df['year_of_loan']
# 平均每年贷款利率
df['mean_interest'] = df['interest'] / df['year_of_loan']
# 总计还款金额
df['all_monthly_pmt'] = df['monthly_payment'] * df['year_of_loan']
# rest_known_zero 剩余公开记录的数量
df['rest_known_dero'] = df['known_dero'] - df['pub_dero_bankrup']
# 除信贷周转余额外，剩余贷款数额
df['rest_loan_recircle_b'] = df['total_loan'] - df['recircle_b']


# ## 数据分箱

# In[10]:


# 不适合太多， 稳定性， 每个分箱的样本数 > 5%
# df.groupby(['industry'])['is_default'].mean()
bin_number = 12
label_list = list(range(bin_number))
df['total_loan_bin'] = pd.qcut(df['total_loan'], bin_number, labels=label_list, duplicates='drop')


# In[11]:


df_internet_f = df[df['is_internet'] == False].copy()
df_internet_t = df[df['is_internet'] == True].copy()

db_train = df_internet_f[df_internet_f['is_default'].notnull()]
db_test = df_internet_f[df_internet_f['is_default'].isnull()]


# In[12]:


# df.select_dtypes(include='O')


# ## 模型训练

# In[13]:


data = pd.concat([db_train, df_internet_t])


# In[14]:


from autogluon.tabular import TabularPredictor

model_autogluon = TabularPredictor(label='is_default',eval_metric='roc_auc', path="./model")
model_autogluon.fit(data, time_limit=500, hyperparameters = {
#                'GBM': {'num_boost_round': 5000},
               'CAT': {'iterations': 2000},
#                'XT': {'n_estimators': 300},
})
model_autogluon.leaderboard()


# In[15]:


# predict
y_pred = model_autogluon.predict_proba(db_test)[1]
result = pd.DataFrame(columns=['id', 'isDefault'])
result['id'] = db_test['loan_id']
result['isDefault'] = y_pred
result.to_csv('/Users/19723/Desktop/baseline_autogluon6.csv')


# In[ ]:


data.drop(['is_default'], axis=1)


# In[ ]:


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
# model_lgb = lgb.LGBMClassifier(n_estimators=500)
xgb_lgb = xgb.XGBClassifier(learning_rate=0.1, n_estimators=150, gamma=0,subsample=0.8, colsample_bytree=0.8, max_depth=7, random_state=2022, enable_categorical=True)
# model_Cat = CatBoostClassifier(iterations=5000, depth=7, learning_rate=0.001, loss_function='Logloss', eval_metric='AUC', logging_level='Verbose', metric_period=50)
# 模型训练
labels = data['is_default']
xgb_lgb.fit(data.drop(['is_default'], axis=1), labels)
y_pred = xgb_lgb.predict_proba(db_test)
result = pd.DataFrame(columns=['id', 'isDefault'])
result['id'] = test['loan_id']
result['isDefault'] = y_pred[:,1]
result


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 银行欺诈检测比赛

# PDF请访问[银行欺诈检测.pdf](https://www.dropbox.com/s/n2jt4tkuvnlkomg/L1%20AI%E5%A4%A7%E8%B5%9B%E4%BF%A1%E6%81%AF%E4%BB%8B%E7%BB%8DV1.5.pptx?dl=0)。

# ## Initialization 

# In[14]:


# 导入包 用于读取文件
import pandas as pd


# 读数据，合并

# In[16]:


# train = pd.read_csv('data/bankfraud_train.csv')
# test = pd.read_csv('data/bankfraud_test.csv')
# 合并train 和 test 集
data = pd.concat([train, test], axis=0)
# 用于其他features文件的合并
# data = pd.merge(data, car_price,on='auto_model', how='outer')


# In[17]:


# 声明一些变量
# 美国大城市，用于feature engineering
top_cities = ["NewYork","LosAngeles","Chicago","Houston","Phoenix","Philadelphia","SanAntonio","SanDiego","Dallas","SanJose","Austin","Jacksonville","FortWorth","Columbus","Indianapolis","Charlotte","SanFrancisco","Seattle","Denver","Washington","Nashville","OklahomaCity","ElPaso","Boston","Portland","LasVegas","Detroit","Memphis","Louisville","Baltimore","Milwaukee","Albuquerque","Tucson","Fresno","Sacramento","KansasCity","Mesa","Atlanta","Omaha","ColoradoSprings","Raleigh","LongBeach","VirginiaBeach","Miami","Oakland","Minneapolis","Tulsa","Bakersfield","Wichita","Arlington"]


# ## view the data

# In[18]:


train['incident_city'].value_counts()


# In[19]:


data['auto_model'].value_counts().count()


# In[20]:


# 用于查看数据中非数字列特殊值的个数
for col in data.select_dtypes(include=object).columns:
    # nqunique是一个方法，记得加括号
	print(col, data[col].nunique())


# ## Feature engineering

# ### 把日期转换成更有意义的特征，列入星期几

# In[21]:


data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'], errors='coerce')
data['incident_date'] = pd.to_datetime(data['incident_date'], errors='coerce')


# In[22]:


data['policy_bind_date_weekday'] = data['policy_bind_date'].dt.weekday
data['incident_date_weekday'] = data['incident_date'].dt.weekday


# In[23]:


base_date = data['policy_bind_date'].min()
data['p_diff'] = (data['policy_bind_date'] - base_date).dt.days
data['i_diff'] = (data['incident_date'] - base_date).dt.days
data.drop(['policy_bind_date', 'incident_date'], axis=1, inplace=True)


# In[24]:


# 日期求差值
data['pi_diff'] = data['p_diff'] - data['i_diff']


# ### 手动对某个特征做one-hot处理
# 

# In[25]:


# t = pd.read_csv('data/bankfraud_train.csv')
# temp = pd.DataFrame(columns=["months_as_customer"])
# temp = t["months_as_customer"] 
# t = temp
# func1 = lambda x: 1 if x == True else 0
# t["months_as_customer"] = t["months_as_customer"]  > 24
# t["months_as_customer"]  = t["months_as_customer"] .apply(func1)
# t


# ### 对某类别数据进行bin处理

# In[26]:


v = pd.DataFrame({
    'top_state': ["NY", "SC", "WV"],
    'second_state': ["NC", "VA", "chi"], 
    'third_state' : ["PA", "OH", "ttt"]
})

data['big_state'] = data['incident_state'].apply(lambda x: '3' if x in v['top_state'].values else '2' if x in v['second_state'].values 
                                             else '1' if x in v['third_state'].values else '0')


# In[27]:


data.drop('incident_state', axis=1, inplace=True)


# In[28]:


v = pd.DataFrame({
    'top_city': ["Springfield", "Arlington", "Columbus"],
    'second_city': ["Northbend", "Hillsdale", "chi"], 
    'third_city' : ["Riverwood", "Northbrook", "ttt"]
})

data['big_city'] = data['incident_city'].apply(lambda x: '3' if x in v['top_city'].values else '2' if x in v['second_city'].values 
                                             else '1' if x in v['third_city'].values else '0')

# 处理好以后可以删了
data.drop('incident_city', axis=1, inplace=True)


# ### 建立特殊值表， 做label encode

# In[29]:


column_name = []
unique_value = []

for col in data.select_dtypes(include=object).columns:
	column_name.append(col)
	unique_value.append(data[col].nunique())


# In[30]:


df = pd.DataFrame()
df['col_name'] =  column_name
df['value'] = unique_value
df = df.sort_values('value', ascending=False)
df


# In[31]:


temp = pd.DataFrame()
cat_columns = data.select_dtypes(include='O').columns
float_d = data.copy()
cat_l = list(cat_columns)
for i in cat_l:
    float_d.drop(i,axis=1, inplace=True)


# In[32]:


from sklearn.preprocessing import LabelEncoder

for col in cat_columns:
	le = LabelEncoder()
	temp[col] = le.fit_transform(data[col])

temp['index'] = range(1, len(temp) + 1)
temp.set_index('index')


# ### 爬虫车牌列unique数据的车价，用车价给各位欠款人做个特征

# In[ ]:


# 简单爬虫代码
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.wait import WebDriverWait
import time
import datetime
import logging
import random
import openpyxl
import pandas as pd
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# 配置浏览器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
wb = openpyxl.Workbook()
sheet = wb.active
sheet.append(['car', 'price'])

# 关闭左上方 Chrome 正受到自动测试软件的控制的提示
options = webdriver.ChromeOptions()
options.add_experimental_option('useAutomationExtension', False)
options.add_experimental_option("excludeSwitches", ['enable-automation'])
browser = webdriver.Chrome(executable_path=chrome_driver, options=options)

# 导入车数据
car = pd.read_excel('./车.xlsx')
car = car["car"].tolist()

# 爬国外二手车网站
def foreignWeb(car):
    chrome_driver = r'./win/chromedriver'
    # 关闭左上方 Chrome 正受到自动测试软件的控制的提示
    options = webdriver.ChromeOptions()
    options.add_argument('--incognito')
    options.add_argument('blink-settings=imagesEnabled=false') # 不載入圖片,提升速度
    options.add_argument('User-Agent=Mozilla/5.0 (Linux; U; Android 8.1.0; zh-cn; BLA-AL00 Build/HUAWEIBLA-AL00) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/57.0.2987.132 MQQBrowser/8.9 Mobile Safari/537.36')

#     options.add_experimental_option('useAutomationExtension', False)
#     options.add_experimental_option("excludeSwitches", ['enable-automation'])
    browser = webdriver.Chrome(executable_path=chrome_driver, options=options)
    browser.get('https://www.carmax.com/cars?search=BMW')
#     wait = WebDriverWait(browser, 20)
    ## 解决弹窗
#     time.sleep(20)
#     ttype = browser.find_element_by_xpath('//button/parent::div[@class="tour-popover-next-button"]')
#     print(ttype)
    
    xxpe = browser.find_element_by_xpath('//div[contains(text(), "close filters.")]')
    print(xxpe)
    time.sleep(20)
#     browser.quit()

# browser.maximize_window()
# 设定最长等待时间  在10s内发现了输入框已经加载出来后就输入“网易云热评墙”
num = 0
for i in car:
    foreignWeb(i)
    num += 1

# 保存数据  输出日志信息  退出浏览器
wb.save(filename='car_info2.xlsx')
logging.info(f'共获取{num}条信息')
browser.quit()


# In[ ]:


# 易车网的爬虫代码
def Chineseprocess(car):
    wait = WebDriverWait(browser, 20)
    _input = wait.until(ec.presence_of_element_located((By.CLASS_NAME, 'yccmp-search-input')))
    # # 搜索框中输入内容，输入之前先清空
    # _input.clear()
    # _input.send_keys('Forrestor')
    # # class定位   模拟点击搜文章
    # browser.find_element_by_xpath("//input[@class='yccmp-search-btn']").click()
    # time.sleep(10)

    _input.clear()
    _input.send_keys(car)
    # class定位   模拟点击搜文章
    browser.find_element_by_xpath("//input[@class='yccmp-search-btn']").click()
    
    try:
        elem = browser.find_element_by_xpath('//*[@class="pp-car-list"]/ul')
        all_li = elem.find_elements_by_tag_name("li")
        for li in all_li:
            text = li.text
            sheet.append([car, text])
    except Exception:
        sheet.append([car,""])
    
    time.sleep(5)


# In[ ]:


browser.get('https://so.yiche.com/chexing/')
# browser.maximize_window()
# 设定最长等待时间  在10s内发现了输入框已经加载出来后就输入“网易云热评墙”
num = 0
for i in car:
    Chineseprocess(i)
    num += 1


# In[ ]:


# 保存数据  输出日志信息  退出浏览器
wb.save(filename='car_info.xlsx')
logging.info(f'共获取{num}条信息')
browser.quit()


# 
# ## 标准化处理的scratch

# In[33]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name == 'model_price' or feature_name == 'policy_number':
            continue
        else:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


# In[34]:


float_d = normalize(float_d)
float_d['index'] = range(1, len(float_d) + 1)
float_d.set_index('index')


# In[35]:


data = pd.merge(temp,float_d,on='index')


# ## 训练模型

# In[10]:


# from sklearn.model_selection import RandomizedSearchCV
# import lightgbm as lgb

# rs_params = {

#         'colsample_bytree': (0.5, 0.6, 1),
#         'learning_rate': (0.005, 0.1, 0.2, 0.3),
#         'reg_lambda': (0.25, 0.3, 0.5, 3, 5),
#         'max_depth': (-1, 2, 3, 5, 10),
#         'min_child_samples': (1, 3, 5, 9, 10),
#         'num_leaves': (20, 2**5-1, 2**5-1, 300, 400),
#         'reg_alpha' : (0.1, 0.25, 0.3, 0.5, 3, 5)
    
# }

# # Initialize a RandomizedSearchCV object using 5-fold CV-
# # 折15次，每次用100样本
# rs_cv = RandomizedSearchCV(estimator=lgb.LGBMClassifier(), param_distributions=rs_params, cv = 7, n_iter=300,verbose=1)

# # Train on training data
# rs_cv.fit(train.drop(['fraud_reported'], axis=1), train['fraud_reported'],verbose=1)
# print(rs_cv.best_params_)
# print(rs_cv.best_score_)


# In[452]:


# model_lgb = lgb.LGBMClassifier(num_leaves = 300, 
#                                reg_alpha=5, 
#                                reg_lambda=0.5, 
#                                objective='binary', 
#                                max_depth=3, 
#                                learning_rate=0.3, 
#                                min_child_samples=5, 
#                                random_state=7777,n_estimators=2000,subsample=1, colsample_bytree=1,)


# In[453]:


# model_lgb.fit(train.drop(['fraud_reported'], axis=1), train['fraud_reported'])


# In[454]:


# y_pred = model_lgb.predict_proba(test.drop(['fraud_reported'], axis=1))


# In[455]:


# train['fraud_reported'].mean()


# In[456]:


# sum(y_pred) / 300


# In[457]:


# result = pd.read_csv('sampleSubmission.csv')
# # 调整矩阵形状
# result['fraud_reported'] = y_pred[:, 1]


# In[460]:


# result


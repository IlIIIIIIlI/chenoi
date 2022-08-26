#!/usr/bin/env python
# coding: utf-8

# (mentalCal)=
# # practice mental calculation

# In[3]:


import random 
import time
import re
import datetime
import pandas as pd
import pymongo
from pymongo import MongoClient
from tqdm import tqdm

def init_currentDate():
    currentDate = datetime.datetime.now().strftime("%Y%m%d")
    return currentDate

# 定义一个插入方法
def insert(test, collection):
    test.reset_index(inplace=True)
    data_dict = test.to_dict("records")
    collection.insert_many(data_dict)

pd.set_option("display.max_rows", None)
client = MongoClient(
    "mongodb+srv://Andyyang:dRk5ZcCl0AzjiWAr@chenoilab.wfhpnl3.mongodb.net/?connect=direct"
)
db = client["Calculation"]
collection = db["Calculation"]

a = random.randrange(11, 99)
b = random.randrange(11, 99)
print(str(a)+"*"+str(b))
time_start=time.time()
print(str(a)+"*"+str(b))
result = input("your answer: ")
result = re.findall("\d+",result)[0]
time_end=time.time()
check = int(result)==int(a*b)
print(check)
print("your input: " + str(result))
print("the true answer: " + str(a*b))
print('time cost: ',time_end-time_start,'seconds')
new_row = pd.DataFrame({'Date':init_currentDate(), 'f_n':a, 's_n':b, 'diff':int(int(result)-int(a*b)), 'check':check, 'time_spend':float(time_end-time_start)}, index=[0])
insert(new_row, collection)


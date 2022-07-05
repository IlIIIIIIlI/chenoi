#!/usr/bin/env python
# coding: utf-8

# (mentalCal)=
# # practice mental calculation

# In[ ]:


import random 

a = random.randrange(11, 99)
b = random.randrange(11, 99)
print(str(a)+"*"+str(b))
result = int(input("your answer: "))
print(result==a*b)


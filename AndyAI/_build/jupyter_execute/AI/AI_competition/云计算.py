#!/usr/bin/env python
# coding: utf-8

# (GPU)=
# 
# # GPU云计算平台

# 笔者相信在看的朋友在入门深度学习之后都会开始接触到GPU，那么GPU能做什么，能做多深，这点互联网上的篇幅数不胜数，大家可以自行了解，笔者在这里不过多阐述。在这篇文章中我主要分享下我踩过的坑，以及我踩坑之后对比出的优胜选项，供大家参考。因此这篇很有实用性，在大赛之前看上一看有助于大家省去大量时间。

# # Cloud Computation Provider

# 笔者用过的云计算平台（不讨论使用深度）有：
# 1. [amazon sagemaker](https://console.aws.amazon.com/sagemaker/home)
# 2. [azure ML](https://portal.azure.com/?quickstart=true#home)
# 3. [OpenBayes](https://openbayes.com/console/chenoi/containers)
# 4. [MistGPU](https://mistgpu.com/create_simple/)
# 5. [恒源云](https://gpushare.com/)
# 6. [矩池云](https://matpool.com/host-market/gpu)
# 7. [autoDL](https://www.autodl.com/console/homepage/personal)
# 
# 笔者将对上面1,2,5,6,7做个人体验评价。

# ## sagemaker
# 
# 如果看官非se或者dev，那么笔者是极度不推荐AWS这个平台的。用户体验而言笔者觉得还不如ali cloud。很多时候明明按照了文档或者油管视频进行操作，到最后发现有错误，且由于不是代码，能获得的只有一个错误的结果，而非解释该错误的过程。且在google上搜索，也不会像stack overflow一样有论坛，而是直接跳转到更多的aws文档中。
# ```{warning}
# AWS文档 -> 出错 -> google -> AWS文档
# ```
# 由于笔者曾经误打误撞进了sagemaker，又因为其不像EC2一样有dashboard，或者提示计费情况，笔者在关闭网页后曾被该网站上过一课。
# ```{image} images/sagemaker.png 
# :name: label
# ```

# ## azure ML
# 微软azure笔者认为是个不错的平台，其最大的特色是automated ML。代码变成拖动的模块，减少了码字的需求，减少了枯燥乏味的数据处理。但是运行费用不明朗。而且笔者个人认为不适合新手小白。这绝对不是一个捷径。
# ```{image} images/azure.png 
# :name: label
# ```

# ## [恒源云](https://gpushare.com/auth/register?user=13*****5074&fromId=3c810081f8a2&source=link)
# 根据其官网的描述，笔者认为很多大佬应该都会用这个平台做学术论文。各方面都很不错，无论是性价比，技术文档，网盘空间还是优惠券，都深得我心。唯一的不足是在VScode上使用，由于IP影响，海外用户会有明显卡顿，如果后期改好了，我肯定还是会用这个平台的。
# ```{image} images/gpushare.png 
# :name: label
# ```

# ## 矩池云
# 这个各方面也不错，只不过价格是上面的两倍。
# ```{image} images/juzhen.png 
# :name: label
# ```

# ## [autoDL](https://www.autodl.com/register?code=5df374a0-a665-41e2-addf-fa6f463a2477)
# 这个是我目前在使用的，我认为各方面都和恒源云特别相似，不得不让人觉得是c-o-p-y的。出现时间比恒源云晚，操作手册基本一样。。但是！海外用户无延迟，且价格，，我认为是全网最低。[温馨提示：新用户送10元优惠券。](https://www.autodl.com/register?code=5df374a0-a665-41e2-addf-fa6f463a2477)
# ```{image} images/autodl.png 
# :name: label
# ```
# 感觉再过一段时间显卡就会被抢爆了。。。

# # 云上trick

# ## 邮件提醒
# 
# 笔者认为大家对于GPU平台的价格应该会是考虑的重中之重，毕竟显卡配置几家也都差不多。在以上这些provider中，恒源云有提供[执行训练并自动上传结果后关机](https://gpushare.com/docs/best_practices/train/#upload%60)的教程。这是因为他们有配套的oss系统，能把运行结果上传到个人网盘，那么对于其他没有这种自动关机的GPU平台来说，要想知道运行进度，好像唯有打印log方法。但是log保存在云上，想实时监控也得登上去看，时刻挂念着运行进度，这真的是“磨人的小妖精”。那么，有什么办法能够使我们定时或者及时知道程序运行完了呢？笔者在实际体验后认为，估算运行时间是一种方法，但是用代码直接发送进度给邮件，感觉这应该会不错，那么实际操作难度如何呢？了解以下几点可使你减少大量时间。
# 
# ### 选择合适
# 要想使用代码发送邮件，首先得了解一个概念，谁来发送？如果我们将邮件发送的代码放在云上，运行的时候，云端执行。因此我们发送邮件（或者访问登录发件人邮件）的位置在于服务器所在地点。根据以上云服务商，中文字的都在国内，因此如果你使用中文的提供商，邮箱最好使用QQ或者网易。如果不是的话，那么google、yahoo或者GXM都是可以的。
# 
# ### 发布流程
# ```{note}
# 获得SMTP服务器 -> 代码登录 -> 获得SMTP协议 -> 发送邮件
# ```
# 看官可以发现流程中我提到的SMTP，全称是“Simple Mail Transfer Protocol”，即简单邮件传输协议。简单地说就是要求必须在提供了账户名和密码之后才可以登录SMTP服务器。一旦我们登录了SMTP服务器，输入App password，然后我们就可以发送邮件了，是不是很简单。因此我们至少需要确定两点。
# 
# - 确定该邮件提供商有SMTP server，以及最好有非SSL端口
# - 确定该邮件有客户端授权密码（app password），这种密码和一般的密码不一样，是用于server登录的。
# 
# 用Yahoo邮箱举例：
# [设定完app password后](https://login.yahoo.com/myaccount/security/)，直接将生成的key放入代码就可以使用
# ```{image} images/yahoo2.png 
# :name: label
# ```
# ````{warning}
# 但是不知什么原因这段时间用不了了，google也是一样，[reddit论坛](https://www.reddit.com/r/yahoo/comments/v5hkc6/yahoo_mail_app_password_not_working/)可以搜索到相关信息。
# ```{image} images/yahoo.png 
# :name: label
# ```
# ````
# 
# 但是没关系，对于我们国内的小伙伴来说，就算他可以使用，服务器也因为一些原因无法访问到这些服务商。我们还是相信伟大的网易。
# 
# - 首先我们先登录[网易云邮箱](https://email.163.com/), 并在设置中进入SMTP选项
# ```{image} images/netease2.png 
# :name: label
# ```
# 
# - 开启服务
# ```{image} images/netease3.png 
# :name: label
# ```
# 
# 这时我们已经接近完成了！

# ### 代码
# 在开码之前，我们把最后一个参数处理了，即服务器端口。由于咱们大部分的GPU云都是本地服务，没有SSL证书，因此我们需要走[25通道](
# https://help.mail.163.com/faqDetail.do?code=d7a5dc8471cd0c0e8b4b8f4f8e49998b374173cfe9171305fa1ce630d7f67ac21b87735d7227c217)，由于笔者之前玩过EC2和SSL证书，所以这一步踩坑比较快。

# In[4]:


def email(contents):        
    # a f**k non-ssl port, you have to use it
    server = smtplib.SMTP('smtp.163.com', 25)
    # server.starttls()
    # login的第二个参数为生成的密钥
    server.login('example@163.com', 'AAISUDHUASGYVB')
    msg = MIMEText(contents, 'plain', 'utf-8')
    msg['Subject'] = Header("email_header_example",'utf-8')
    server.sendmail('example@163.com', 'receiver_example@gmail.com', msg.as_string())
    server.quit()


# 怎么样，是不是很简单。笔者为实现该功能花了五小时。。。这主要还是受到了google 5月30日 新政策的影响。罢了罢了。。。

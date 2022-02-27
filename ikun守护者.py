#!/usr/bin/env python
# coding: utf-8

# # 【飞桨领航团AI创造营】 基于PaddleX的路面危险警告器
# <font size="2" color="blue" font weight="bolder">基于PaddleX的一套路面危险警告器，旨在帮助那些走路不注意脚下的低头族避免因沉迷手机而导致不同程度的跌倒，同时也可用于老人与小孩行走过程中注意前方路面</font>

# # 1、项目背景
#            偶尔在电视上看到一些新闻，说啥，有人走路只顾着没低头看手机，没注意前方的路面情况，结果摔成重伤住院，还有的直接掉进下水道。看到这样新闻，我想问：兄得，你在干什么？看到这样悲剧的发生，我同情心泛滥，我决定。你们的安全将由我来守护。于是，一个基于路面的情况的报警器的想法就出生了。有了想法，就得行动。那么首先得找个数据集，是吧，这个数据集还得是路面的。我就在网上找了找.哎，在AI Studio上真找到了一个非常不错的路面数据集，天助我也。
#           并且听说Paddle训练模型很容易，而且还有免费的GPU算力白嫖，于是，我果断决定在AI Studio上面开我的守护模型训练之旅。
#           ikun们，你们的安全，将由我来守护。
#           （PS:这个项目其实不至于针对低头族，还可以使用老人与小孩，助他们安全出行）

# # 2、数据集介绍及预处理

# ## 2.1 数据简介
#   那么我在平台找到的数据集 [【全球开放数据创新应用大赛】](https://aistudio.baidu.com/aistudio/datasetdetail/93683)的比赛数据集，简直是爱了爱了，该数据集里面包含6000张图片以及一个包含所有图片的标注信息的json文件，这个json文件比较难搞，不处理一下不利于我们后面的操作，万幸，有大佬用过这个数据集，参考他的处理方式，我将整个数据集处理成了VOC格式 ，便于后续操作，我已经将转换好的数据集上传，就是本项目挂载的。 

# ## 2.2 解压数据集并查看

# In[11]:


get_ipython().system('pwd')


# In[20]:


get_ipython().system('unzip -oq data/data129284/VOC.zip')


# In[1]:


#数据集结构
get_ipython().system('tree home/aistudio/dataset/VOC -d')


# In[3]:


#查看数据集图片
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(cv2.imread('home/aistudio/dataset/VOC/JPEGImages/00001.jpg'))
plt.tight_layout()
plt.show()


# ## 2.3 安装PaddleX及依赖
# 

# In[11]:


get_ipython().system(' pip install "paddlex<=2.0.0"')
get_ipython().system(' pip install scikit-image ')
get_ipython().system(' pip install threadpoolctl==2.0.0 -i https://mirror.baidu.com/pypi/simple')
get_ipython().system(' pip install scikit-learn==0.23')
get_ipython().system('pip install paddle2onnx')


# ## 2.4 划分数据集

# In[ ]:


get_ipython().system('paddlex --split_dataset --format VOC --dataset_dir home/aistudio/dataset/VOC --val_value 0.2 --test_value 0.1')


# ```
# 2022-02-24 10:28:17 [INFO]	Dataset split starts...
# 2022-02-24 10:28:18 [INFO]	Dataset split done.
# 2022-02-24 10:28:18 [INFO]	Train samples: 4200
# 2022-02-24 10:28:18 [INFO]	Eval samples: 1200
# 2022-02-24 10:28:18 [INFO]	Test samples: 600
# 2022-02-24 10:28:18 [INFO]	Split files saved in home/aistudio/dataset/VOC
# ```

# ## 3、模型训练

# ## 配置GPU

# In[25]:


import matplotlib
matplotlib.use('Agg') 
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx


# ## 3.1 定义训练和验证时的transforms

# In[35]:


from paddlex import transforms
import paddlex as pdx

train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(), transforms.Resize(
        target_size=608, interp='RANDOM'), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'), transforms.Normalize()
])


# ## 3.2 定义训练和验证所用的数据集

# In[41]:


train_dataset = pdx.datasets.VOCDetection(
    data_dir='home/aistudio/dataset/VOC',
    file_list='home/aistudio/dataset/VOC/train_list.txt',
    label_list='home/aistudio/dataset/VOC/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='home/aistudio/dataset/VOC',
    file_list='home/aistudio/dataset/VOC/val_list.txt',
    label_list='home/aistudio/dataset/VOC/labels.txt',
    transforms=eval_transforms,
    shuffle=False)


# In[48]:


# 初始化模型 并训练
num_classes = len(train_dataset.labels)

model = pdx.det.PPYOLO(num_classes=num_classes)


# ## 3.3 启动训练

# In[52]:


model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_interval_epochs=1,
    lr_decay_epochs=[1,10],
    save_dir='output/ppyolo',
    )


# ## 4、模型导出及预测

# In[ ]:


get_ipython().system('paddlex --export_inference --model_dir=output/ppyolo/best_model --save_dir=./inference_model ')


# ```
# 2022-02-24 14:34:51 [INFO]	The model for the inference deployment is saved in ./inference_model/inference_model.
# ```

# In[ ]:


#进行单个图片预测
import paddlex as pdx
model = pdx.load_model('inference_model/inference_model')
result = model.predict('00007.jpg')
print(result)
pdx.det.visualize('00007.jpg', result, threshold=0.09, save_dir='./')


# ```
# 2022-02-24 15:19:53 [INFO]	Model[PPYOLO] loaded.
# [{'category_id': 1, 'category': 'Manhole', 'bbox': [879.882080078125, 949.22021484375, 216.2427978515625, 63.9039306640625], 'score': 0.010250603780150414}]
# 2022-02-24 15:19:54 [INFO]	The visualized result is saved at ./visualize_00007.jpg
# ```

# In[67]:


get_ipython().run_line_magic('matplotlib', 'inline')
 
import cv2
import pandas as pd
import matplotlib.pyplot as plt
imgpath = 'visualize_00007.jpg'
print(imgpath)
image = cv2.imread(imgpath)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 


# ## 5、总结

# 以上就是模型训练的全过程了，其实是有些坑的0。整个的训练也就练了10轮，可以看出模型不咋行，还有改进空间，主要是为了赶上创造营结营，潦草了点。
# 后期规划：肯定是要加大训练轮次的，并进行参数的调优。模型练好后，也要开始进行落地部署了，希望守护最好的你能够部署成功，我也不留遗憾。好了，就说这么多。最后鸣谢大佬们的帮助。

# 个人介绍 余卓凡 飞桨社区小菜鸡一枚 欢迎给我点点关注
# 
# 我在AI Studio上获得白银等级，点亮2个徽章，来互关呀~ https://aistudio.baidu.com/aistudio/personalcenter/thirdview/879971

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

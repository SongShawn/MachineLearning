
# coding: utf-8

# In[1]:


import pandas as pd
import visual as vs
import numpy as np
import importlib
import matplotlib.pyplot as plt
import preprocess as datapre

importlib.reload(vs)
importlib.reload(datapre)


# ## 数据探索
# 来自于Kaggle名为旧金山罪案类型分类的数据集，该数据集分为训练集和测试集，训练集包含878049个带标签样本，测试集包含884262个未带标签样本。  
# 运行下边代码加载训练集和测试集。

# In[2]:


train_data = pd.read_csv('../datasets/train.csv', parse_dates=['Dates'])
test_data = pd.read_csv('../datasets/test.csv', parse_dates=['Dates'])


# ### 显示训练集部分样本

# In[3]:


train_data.head(4)


# ### 显示测试集部分样本

# In[4]:


test_data.head(1)


# ### 删除训练集中无用字段
# 训练集中的'Descript'，'Resolution'两个属性无意义，直接删除

# In[5]:


# 删除无用字段 'Descript' 'Resolution'
train_data = train_data.drop(columns=['Descript', 'Resolution'])


# ### 处理数据集中的缺失值

# In[6]:


train_data.isnull().any(axis=0)


# In[7]:


train_data.isna().any(axis=0)


# **确定训练集中没有缺失数据。**

# ### 特征提取

# #### Dates解析为年月日时

# In[8]:


train_data = datapre.extra_dates(train_data)


# #### DayOfWeek转化为数字

# In[9]:


from sklearn.preprocessing import LabelEncoder
DayOfWeekEnc = LabelEncoder()
train_data['DayOfWeekID'] = DayOfWeekEnc.fit_transform(train_data['DayOfWeek'])


# #### PdDistrict转化为数字

# In[10]:


PdDistrictEnc = LabelEncoder()
train_data['PdDistrictID'] = PdDistrictEnc.fit_transform(train_data['PdDistrict'])


# #### 从Address中提取是否含有Block字段作为特征

# In[11]:


train_data = datapre.extra_address_for_block(train_data)


# #### 从Address中提取地址后缀作为特征
# - 标准的地址都会含有一个简写后缀表示道路的类型，直接解析后缀，如："200 Block of INTERSTATE80 HY"解析为"HY"。
# - 对于路口则会表示成'XX ST / YY ST'，直接解析为"CrossRoad"，如："STCHARLES AV / 19TH AV"解析为"CrossRoad"。
# - 对于直接含有道路类型全名的地址也要进行解析。如："0 Block of AVENUE OF THE PALMS"中的"AVENUE"就是道路类型。
# - 对于上述三种方式都无法解析，则直接设置为"Unkown"。  
# 
# 根据直觉判断，不同类型的道路发生各种类型犯罪的分布是不一样的。如：铁路附近发生自杀案件的概率普遍高于其他案件。

# In[12]:


train_data['PositionType'] = None
suffixs = ["AL","AV","BL","CR","CT","DR","EX","HWY","HY","LN","PL","PZ","RD","ST","TR","WY","WAY"]
suffix_names = ["Alley","Avenue","Boulevard","Circle","Court","Drive","Expressway","Highway","Highway",
                "Lane","Place","Plaza","Road","Street","Terrace","Way","Way"]
cross_road = "CrossRoad" # 交叉路口，含有/的
unkown_road = "Unkown"

# 设置交叉路口
train_data.loc[train_data['Address'].str.contains('/'), 'PositionType'] = cross_road

# 查找缩写并设置
for a in suffixs:
    train_data.loc[(train_data['Address'].str.contains('/') == False) 
                   & (train_data['Address'].str.contains(' '+a+'$')), 'PositionType'] = a
    
# 查找全程并设置
for i,d in enumerate(suffix_names):
    train_data.loc[(train_data['PositionType'].isna())
                   & (train_data['Address'].str.contains(d, case=False)), 'PositionType'] = suffixs[i]
    
# 无法解析的均设置为Unkown
train_data.loc[(train_data['PositionType'].isna()), 'PositionType'] = unkown_road

# 合并 HWY HY，合并 WY WAY
train_data.loc[train_data['PositionType'] == "HWY", "PositionType"] = "HY"
train_data.loc[train_data['PositionType'] == "WAY", "PositionType"] = "WY"


# #### PositionType转化为数字

# In[13]:


PositionTypeEnc = LabelEncoder()
train_data['PositionTypeID'] = PositionTypeEnc.fit_transform(train_data['PositionType'])


# In[14]:


train_data.columns


# #### 保存预处理后的训练集

# In[15]:


train_data.to_csv("../datasets/train_preprocess.csv", index=False)


# #### 删除含有异常值的样本
# 属性'Category'、'PdDistrict'为类别，存在既有意义。属性'Address'为自由字符串，存在既有意义。
# - Dates，需要在特定区间内
# - DayOfWeek，只能有七种类型
# - X，Y，

# In[16]:


X_bound = [-122.52, -122.35]
Y_bound = [37.70, 37.85]


# In[17]:


train_data.shape


# In[18]:


train_data.loc[(train_data['X'] > X_bound[0]) & (train_data['X'] < X_bound[1]) 
               & (train_data['Y'] > Y_bound[0]) & (train_data['Y'] < Y_bound[1])].shape


# #### 设置训练模型特征名字

# In[17]:


feature_names = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID']


# ### 使用KNN算法进行分类
# 使用KNN算法以X、Y经纬度作为特征对训练集进行分类，将分类结果作为新的特征并入到训练集中。

# ### 使用XGBoost作为预测算法

# In[18]:


# 取出训练集中的预测目标字段

sample_data = train_data.sample(frac = 0.3, random_state=10)

target = sample_data['Category']
features = sample_data[feature_names]


# #### 开始训练模型

# In[19]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[20]:


LabelEncTarget = LabelEncoder()
target = LabelEncTarget.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print('X_train has {} samples.'.format(X_train.shape[0]))
print('X_test has {} samples.'.format(X_test.shape[0]))

DTrain_X = xgb.DMatrix(data=X_train, label=y_train)
DTest_X = xgb.DMatrix(data=X_test, label=y_test)


# In[21]:


from itertools import product

def products(data):
    results = [[]]
    for a in data:
        results = [x+[y] for x in results for y in a]
    for a in results:
        yield tuple(a)
        
def extra_params(param, param_grid):
    param_names = list(param_grid)
    params = []
    filenames = []
    param_sets = products(param_grid.values())
    for a in param_sets:
        b = param.copy()
        for key,value in zip(param_names, a):
            b[key] = value
        params.append(b)
        filenames.append('_'.join([str(i) for i in a]))
    return params,filenames


# In[ ]:


import time
import datetime
from os import makedirs, system

param = {}
param['objective'] = 'multi:softprob'
# scale weight of positive examples
# param['eta'] = 0.01
# param['max_depth'] = 6
# param['silent'] = 1
# param['nthread'] = 8
param['num_class'] = len(LabelEncTarget.classes_)

param['eval_metric'] = 'mlogloss'
param['tree_method'] = 'gpu_hist'

evallist = [(DTrain_X, 'train'), (DTest_X, 'Test')]

param_grid = {
#     'eta' : [0.01],
    'max_depth': [9],
    'subsample' : [0.7],
#     'grow_policy' : ["depthwise", "lossguide"],
}

num_round = 1000

# 参数和模型保存文件名
params,files = extra_params(param, param_grid)

# 模型保存文件路径
model_path = "../models/"+str(datetime.datetime.utcnow()).replace(':','_').replace(' ', '_').replace('.', '_') + "_" + str(num_round)
makedirs(model_path)

# 模型文件名的含义
modelfileformat = "-".join(list(param_grid))
system("echo 123 > " + model_path + "/" + modelfileformat)
 
start = time.time()
result_data = {'best_iteration' : [],
               'best_score' : []}

def eta_calc(round_index, round_count):
    print("eta calc: ", round_index, round_count)
    etas = [0.1, 0.07, 0.05, 0.01]
    for i in range(1,1+len(etas)):
        if round_index < round_count/len(etas)*i:
            return etas[i-1]
    return etas[-1]

for i,a in enumerate(params):
    bst =xgb.train(a, DTrain_X, num_round, evallist, early_stopping_rounds=None, learning_rates=eta_calc)
    bst.save_model(model_path + "/" + files[i] + ".model")
    result_data['best_iteration'].append(bst.best_iteration)
    result_data['best_score'].append(bst.best_score)

results = pd.DataFrame(index=files, data=result_data)
results.to_csv(model_path + "/" + "result.csv")

print('GPU Training Time: %s seconds.' % (str(time.time() - start)))


# In[ ]:


# print(bst.attributes())


# # #### 预测验证集

# # In[ ]:


# test_data = datapre.extra_dates(test_data)
# test_data['DayOfWeekID'] = DayOfWeekEnc.transform(test_data['DayOfWeek'])
# test_data['PdDistrictID'] = PdDistrictEnc.transform(test_data['PdDistrict'])
# test_data = datapre.extra_address_for_block(test_data)
# test_data = datapre.extra_address_for_suffix(test_data)


# # In[ ]:


# test_data['PositionType'] = PositionTypeEnc.transform(test_data['PdDistrict'])
# X_valid = test_data[feature_names]
# DX_valid = xgb.DMatrix(data=X_valid)


# # In[ ]:


# y_pred_prob = np.round(bstgpu.predict(DX_valid), 4)


# # In[ ]:


# from sklearn.metrics import  log_loss, accuracy_score


# # In[ ]:


# log_loss(y_test, y_pred_prob, labels=xgbclf.classes_)


# # In[ ]:


# accuracy_score(y_test, y_pred)


import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import datetime
from os import makedirs, system


# 取出训练集中的预测目标字段
feature_names = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']
train_data = pd.read_csv("../datasets/train_preprocess.csv")
# sample_data = train_data.sample(frac = 0.3, random_state=10)
sample_data = train_data

target = sample_data['Category']
features = sample_data[feature_names]


# #### 开始训练模型
LabelEncTarget = LabelEncoder()
target = LabelEncTarget.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print('X_train has {} samples.'.format(X_train.shape[0]))
print('X_test has {} samples.'.format(X_test.shape[0]))

DTrain_X = xgb.DMatrix(data=X_train, label=y_train)
DTest_X = xgb.DMatrix(data=X_test, label=y_test)

param = dict()
param['objective'] = 'multi:softprob'
param['silent'] = 0
param['num_class'] = len(LabelEncTarget.classes_)

param['eval_metric'] = 'mlogloss'
param['verbosity'] = 3
param['seed'] = 10

param['max_depth'] = 8
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['tree_method'] = 'gpu_hist'
param['gamma'] = 0.1
param['scale_pos_weight'] = 1


evallist = [(DTrain_X, 'train'), (DTest_X, 'Test')]


num_round = 5000

start = time.time()


bst =xgb.train(a, DTrain_X, num_round, evallist, 
                early_stopping_rounds=20)
# bst.save_model(model_path + "/" + files[i] + ".model")
    

print('GPU Training Time: %s seconds.' % (str(time.time() - start)))
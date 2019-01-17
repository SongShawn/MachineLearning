import pandas as pd
import time
import datetime
import numpy as np

# 取出训练集中的预测目标字段
feature_names = ['Year', 'Month', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']

train_data = pd.read_csv("../datasets/train_preprocess.csv")
# sample_data = train_data.sample(frac = 0.01, random_state=10)
sample_data = train_data

target = sample_data['Category']
features = sample_data[feature_names]

# #### 开始训练模型
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

LabelEncTarget = LabelEncoder()
target = LabelEncTarget.fit_transform(target)

X_train, X_test, y_train, y_test = \
    train_test_split(features, target, test_size=0.2, random_state=42)
print('X_train has {} samples.'.format(X_train.shape[0]))
print('X_test has {} samples.'.format(X_test.shape[0]))

DTrain_X = xgb.DMatrix(data=X_train, label=y_train)
DTest_X = xgb.DMatrix(data=X_test, label=y_test)

param = dict()
param['objective'] = 'multi:softprob'
param['silent'] = 1
param['num_class'] = len(LabelEncTarget.classes_)

param['eval_metric'] = 'mlogloss'
# param['verbosity'] = 4
param['seed'] = 10

param['max_depth'] = 6
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['tree_method'] = 'hist'
param['gamma'] = 2
param['min_child_weight'] = 10
param['max_delta_step'] = 2


evallist = [(DTrain_X, 'train'), (DTest_X, 'Test')]

num_round = 3000
early_stop = 5

etas = [0.1]
start = time.time()
model_name = "../models/model_eta_0.3.model"
bst = None
for i,eta in enumerate(etas) :
    param['eta'] = eta
    bst = xgb.train(param, DTrain_X, num_boost_round=num_round, 
                evals = evallist, early_stopping_rounds=early_stop,
                xgb_model=None)
    model_name = '../models/model_eta_0.01_3000_' + str(eta) + '.model'
    bst.save_model(model_name)
    print('{}.th traning, eta: {}, best_score: {}'.format(i, eta, bst.best_score))

print('CPU hist Trainig Time: %s seconds.' % (str(time.time() - start)))


# predict
start = time.time()
valid_data = pd.read_csv("../datasets/test_preprocess.csv")

X_valid = valid_data[feature_names]
DX_valid = xgb.DMatrix(data=X_valid)

y_pred_prob = np.round(bst.predict(DX_valid), 4)

csv_output = pd.DataFrame(columns=LabelEncTarget.classes_, data=y_pred_prob)
csv_output.insert(0, 'Id', valid_data['Id'])
csv_output.to_csv('../results/model_eta_0.01_3000_result.csv', index=False)

print('CPU Predict Time: %s seconds.' % (str(time.time() - start)))

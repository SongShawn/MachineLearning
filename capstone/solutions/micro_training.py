import pandas as pd

# 取出训练集中的预测目标字段
feature_names = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']

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

model_path = '../models/2019-01-14_07_53_12_277612_5000/0.2_8_0.8_0.8_gpu_hist_0.1_1.model'
# for i,eta in enumerate(etas) :
#     param['eta'] = eta
#     bst = xgb.train(param, DTrain_X, num_boost_round=num_round, 
#                     evals = evallist, early_stopping_rounds=20,
#                     xgb_model = model_path)
#     bst.save_model('../models/temp_' + str(i) + '.model')
#     print('eta: {}, best_score: {}', eta, bst.best_score)

param['eta'] = 0.01
# etas = [0.1 for i in range(num_round)]
bst = xgb.train(param, DTrain_X, num_boost_round=num_round,
                evals=evallist, early_stopping_rounds=50, learning_rates=None)

bst.save_model('../models/ok.model')
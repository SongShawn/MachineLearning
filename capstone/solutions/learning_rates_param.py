import pandas as pd

# 取出训练集中的预测目标字段
feature_names = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']

train_data = pd.read_csv("../datasets/train_preprocess.csv")
sample_data = train_data.sample(frac = 0.01, random_state=10)
# sample_data = train_data

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
param['silent'] = 1
param['num_class'] = len(LabelEncTarget.classes_)

param['eval_metric'] = 'mlogloss'
# param['verbosity'] = 0
param['seed'] = 10

param['max_depth'] = 8
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['tree_method'] = 'hist'
param['gamma'] = 0.1
param['scale_pos_weight'] = 1


evallist = [(DTrain_X, 'train'), (DTest_X, 'Test')]

num_round = 100
early_stop = 1

# etas = [0.2, 0.1, 0.05, 0.01, 0.001]
etas = [0.1 for i in range(num_round)]

print(etas)
# param['learning_rate'] = 0.1
bst = xgb.train(param, DTrain_X, num_boost_round=num_round, 
            evals = evallist, early_stopping_rounds=None,
            learning_rates=etas)

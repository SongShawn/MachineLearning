import pandas as pd

# 取出训练集中的预测目标字段

feature_names = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']


train_data = pd.read_csv("../datasets/train_preprocess.csv")
# sample_data = train_data.sample(frac = 0.3, random_state=10)
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


import time
import datetime
from os import makedirs, system

param = {}
param['objective'] = 'multi:softprob'
# scale weight of positive examples
# param['eta'] = 0.01
# param['max_depth'] = 6
param['silent'] = 0
# param['nthread'] = 8
param['num_class'] = len(LabelEncTarget.classes_)

param['eval_metric'] = 'mlogloss'
# param['tree_method'] = 'gpu_exact'
param['verbosity'] = 0
param['seed'] = 10

evallist = [(DTrain_X, 'train'), (DTest_X, 'Test')]

param_grid = {
    'eta' : [0.2],
    'max_depth': [8],
    'subsample' : [0.8],
#     'grow_policy' : ["depthwise", "lossguide"],
    'colsample_bytree': [0.8],
    'tree_method' : ['gpu_hist'],
    'gamma': [0.1],
    'scale_pos_weight':[1],
}

num_round = 5000

# 参数和模型保存文件名
params,files = extra_params(param, param_grid)

# 模型保存文件路径
model_path = "../models/" \
    + str(datetime.datetime.utcnow()).replace(':','_').replace(' ', '_').replace('.', '_') \
    + "_" + str(num_round)
makedirs(model_path)

# 模型文件名的含义
modelfileformat = "-".join(list(param_grid))
system("echo 123 > " + model_path + "/" + modelfileformat)
 
start = time.time()
result_data = {'best_iteration' : [],
               'best_score' : []}

def eta_calc(round_index, round_count):
    # etas = [0.01, 0.005, 0.001]
    # for i in range(1,1+len(etas)):
    #     if round_index < round_count/len(etas)*i:
    #         return etas[i-1]
    # return etas[-1]
    return 0.4

etas = [0.4 for i in range(num_round)]

for i,a in enumerate(params):
    bst =xgb.train(a, DTrain_X, num_round, evallist, 
                    early_stopping_rounds=5, learning_rates=None)
    bst.save_model(model_path + "/" + files[i] + ".model")
    result_data['best_iteration'].append(bst.best_iteration)
    result_data['best_score'].append(bst.best_score)

results = pd.DataFrame(index=files, data=result_data)
results.to_csv(model_path + "/" + "result.csv")

print('GPU Training Time: %s seconds.' % (str(time.time() - start)))
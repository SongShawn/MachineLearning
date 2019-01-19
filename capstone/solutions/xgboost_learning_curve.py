
# coding: utf-8



import pandas as pd
import time
import datetime
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.metrics import log_loss, make_scorer
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder




# 取出训练集中的预测目标字段
feature_names = ['Year', 'Month', 'Hour', 'DayOfWeekID', 'PdDistrictID', 'HasBlock', 'PositionTypeID', 'X', 'Y']


origin_data = pd.read_csv("../datasets/train_preprocess.csv")


# # 去除样本个数小于1000目标分类
# a = origin_data["Category"].value_counts()
# cate_names = a[a<1000].index.tolist()
# print(cate_names)
# for cate in cate_names:
#     rmv_index = origin_data[origin_data["Category"] == cate].index.tolist()
#     origin_data = origin_data.drop(index=rmv_index)



X = origin_data[feature_names]



target_label = origin_data['Category']
target_enc = LabelEncoder()
y_true = target_enc.fit_transform(target_label)

xgbclf = XGBClassifier(max_depth=6,
                       learning_rate=0.1, 
                       n_estimators=1, 
                       objective="multi:softprob",
                       n_job=-1,
                       gamma=2,
                       min_child_weight=10,
                       max_delta_step=2,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       random_state=10
                    )



train_size, train_score, test_score = learning_curve(estimator=xgbclf,
      X=X, y=target_label, verbose=1, shuffle = True, train_sizes=[0.01],
      cv=10,scoring=make_scorer(log_loss, needs_proba=True), n_jobs=-1, random_state=20)

print(train_size)
print(train_score)
print(test_score)
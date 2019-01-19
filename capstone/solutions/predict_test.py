import time
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

feature_names = ['Year', 'Month', 'Hour', 'DayOfWeekID', 'PdDistrictID', \
                 'HasBlock', 'PositionTypeID', 'RoadBlockID', 'RoadName1ID', 'RoadName2ID', 'X', 'Y']

start = time.time()
valid_data = pd.read_csv("../datasets/test_preprocess.csv")
train_data = pd.read_csv("../datasets/train_preprocess.csv")

TargetEnc = LabelEncoder()
TargetEnc.fit(train_data["Category"])

bst = xgb.Booster()
bst.load_model("../models/model_eta_0.3.model")

X_valid = valid_data[feature_names]
DX_valid = xgb.DMatrix(data=X_valid)

y_pred_prob = np.round(bst.predict(DX_valid), 4)

csv_output = pd.DataFrame(columns=TargetEnc.classes_, data=y_pred_prob)
csv_output.insert(0, 'Id', valid_data['Id'])
csv_output.to_csv('../results/'+"new_add_feature"+'.csv', index=False)

print('CPU Predict Time: %s seconds.' % (str(time.time() - start)))
import pandas as pd
import numpy as np
import time
import preprocess as datapre

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

origin_data = pd.read_csv("../datasets/train_preprocess.csv")

# train_data = datapre.dataset_sample(origin_data, frac=0.5)

feature_names = ['Year', 'Month', 'Hour', 'DayOfWeekID', 'PdDistrictID', \
                 'HasBlock', 'RoadTypeID', 'RoadBlockID', 'RoadName1ID', 'RoadName2ID', 'X', 'Y']

# X = train_data[feature_names]
# y_true = train_data["Category"]


def neg_log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None):
    return -log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

call_neg_log_loss = make_scorer(neg_log_loss, needs_proba=True)

base_params = dict(n_estimators=500, max_depth=5, min_samples_split=20,
                    max_features=0.5, n_jobs=-1, random_state=42)


base_params["criterion"] = "entropy"
base_params["min_samples_split"] = 60
base_params["max_features"] = 0.6
base_params["bootstrap"] = True
base_params["max_depth"] = 10
print(base_params)

total_X = origin_data[feature_names]
total_y_true = origin_data["Category"]

totalTargetEnc = LabelEncoder()
total_y_true = totalTargetEnc.fit_transform(total_y_true)

param_grid = {"n_estimators": list(range(1000,4001,500))}

# base_params["bootstrap"] = cv_clf_bootstrap.best_params_["bootstrap"]
rfclf = RandomForestClassifier(**base_params)
cv_clf_final = GridSearchCV(estimator=rfclf, param_grid=param_grid, 
                                scoring=call_neg_log_loss, 
                                n_jobs=-1, cv=5, verbose=4, return_train_score=True, refit=True)

start = time.time()
cv_clf_final.fit(total_X, total_y_true)
print("Training needs %d seconds." % (time.time()-start))

print(cv_clf_final.best_params_)

print(cv_clf_final.cv_results_)

base_params["n_estimators"]=cv_clf_final.best_params_["n_estimators"]
print(base_params)

best_rf_clf = cv_clf_final.best_estimator_

valid_data = pd.read_csv("../datasets/test_preprocess.csv")
valid_X = valid_data[feature_names]

y_pred_prob = np.round(best_rf_clf.predict_proba(valid_X), 4)

csv_output = pd.DataFrame(columns=totalTargetEnc.classes_, data=y_pred_prob)
csv_output.insert(0, 'Id', valid_data['Id'])
csv_output.to_csv('../results/RandomForestClf_best.csv', index=False)
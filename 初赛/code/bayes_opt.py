import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm as lgb

test = pd.read_csv('test_feature.csv')

test_col = list(test.columns)
print(test.head())

df_good = pd.read_csv('df_good_feature.csv')
good_col = list(df_good.columns)

df_bad = pd.read_csv('df_bad_feature.csv')
bad_col = list(df_bad.columns)

print('test_col', len(test_col))

col_count = 0
for i in test_col:
    if i in good_col:
        col_count+=1
print(col_count)

col_count = 0
for i in test_col:
    if i in bad_col:
        col_count+=1
print(col_count)

# 列交集

use_col = set(test_col) & set(good_col) & set(bad_col)
print(len(use_col))

test['label'] = 'test'

df = pd.concat([df_good,df_bad,test], axis=0, ignore_index=True)
print(df.shape)

df.replace(np.nan, 0, inplace=True)


train_df = df[df['label']!='test'].copy().reset_index(drop=True)
def label2num(x):
    if x == 'good':
        return 0
    else:
        return 1
train_df['label'] = train_df['label'].map(label2num)
train_df = train_df.select_dtypes(exclude='object')
Y_train = train_df['label'].values
print(train_df['label'])
print(train_df.shape)
train_df.drop(columns='label', inplace=True)
train_df = train_df.values
print(train_df.shape)

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold,KFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# y_pred = np.zeros(len(x_test))

# for fold, (train_index, val_index) in enumerate(kf.split(train_df, Y_train)):
#     x_train, x_val = train_df[train_index], train_df[val_index]
#     y_train, y_val = Y_train[train_index], Y_train[val_index]
#     train_set = lgb.Dataset(x_train, y_train)
#     val_set = lgb.Dataset(x_val, y_val)
#
#     params = {
#         'boosting_type': 'gbdt',
#         'metric': {'binary_logloss', 'auc'},
#         'objective': 'binary',  # regression,binary,multiclass
#         'seed': 666,
#         'num_leaves': 20,
#         'learning_rate': 0.1,
#         'max_depth': 6,
#         'n_estimators': 5000,
#         'lambda_l1': 1,
#         'lambda_l2': 10,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 1,
#         'colsample_bytree': 0.9,
#         'verbose': -1,
#     }
#
#     model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50,
#                       valid_sets=[val_set], verbose_eval=100)
#     y_pred += model.predict(x_test, num_iteration=model.best_iteration) / kf.n_splits
# y_pred = [1 if y > 0.5 else 0 for y in y_pred]
# rmse = metrics.accuracy_score(y_pred, y_test)
# print(rmse)


def cv_lgm(num_leaves, max_depth, lambda_l1, lambda_l2, bagging_fraction, bagging_freq, colsample_bytree):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
    accuracy = 0

    for fold, (train_index, val_index) in enumerate(kf.split(train_df, Y_train)):
        x_train, x_val = train_df[train_index], train_df[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        train_set = lgb.Dataset(x_train, y_train)
        val_set = lgb.Dataset(x_val, y_val)

        params = {
            'boosting_type': 'gbdt',
            'metric': {'binary_logloss', 'auc'},
            'objective': 'binary',  # regression,binary,multiclass
            #         'num_class':2,
            'seed': 666,
            'num_leaves': int(num_leaves),  # 20
            'class_weight': 'balanced',
            'learning_rate': 0.1,
            'max_depth': int(max_depth),
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': int(bagging_freq),
            'colsample_bytree': colsample_bytree,
            'min_data_in_leaf': 50,
            'verbose': -1,
        }

        model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=20,
                          valid_sets=[val_set], verbose_eval=20)
        y_pred = model.predict(x_val, num_iteration=model.best_iteration)
        y_pred = [1 if y > 0.5 else 0 for y in y_pred]
        accuracy += metrics.fbeta_score(y_val, y_pred, average='binary', beta=0.5) / kf.n_splits

    return accuracy


from bayes_opt import BayesianOptimization

rf_bo = BayesianOptimization(
    cv_lgm,
    {'num_leaves': (10, 100),
     'max_depth': (3, 15),
     'lambda_l1': (0.1, 50),
     'lambda_l2': (0.1, 50),
     'bagging_fraction': (0.6, 1),
     'bagging_freq': (1, 5),
     'colsample_bytree': (0.6, 1),
     }
)
rf_bo.maximize(n_iter=500)
print(rf_bo.max)

# meiyou balanced
# {'target': 0.9618532397685403, 'params': {'bagging_fraction': 1.0, 'bagging_freq': 1.0, 'colsample_bytree': 1.0, 'lambda_l1': 0.1, 'lambda_l2': 4.8413702888373935, 'max_depth': 15.0, 'num_leaves': 29.5581434829485}}

# with balanced
# {'target': 0.9598589411419569, 'params': {'bagging_fraction': 1.0, 'bagging_freq': 5.0, 'colsample_bytree': 1.0, 'lambda_l1': 0.1, 'lambda_l2': 0.1, 'max_depth': 3.0, 'num_leaves': 26.877978158360712}}

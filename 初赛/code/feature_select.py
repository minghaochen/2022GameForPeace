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

# import re
# df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# select only the numerical features
df_train = df[df['label']!='test'].copy().reset_index(drop=True)
def label2num(x):
    if x == 'good':
        return 0
    else:
        return 1
df_train['label'] = df_train['label'].map(label2num)
df_train = df_train.select_dtypes(exclude='object')


df_test = df[df['label']=='test'].copy().reset_index(drop=True)
df_test['label'] = -1
df_test = df_test.select_dtypes(exclude='object')


count = 0
col_dict = {}
for col in list(df.columns):
    if col != 'label':
        col_dict[col] = f'col_{count}'
        count += 1

df_train = df_train.rename(columns=col_dict)
df_test = df_test.rename(columns=col_dict)


from autox.autox_competition.feature_selection import AdversarialValidation
adversarialValidation = AdversarialValidation()


Id = []
target = 'label'
adversarialValidation.fit(train = df_train, test = df_test, id_ = Id, target = target, categorical_features = [], p = 0.55)

print(adversarialValidation.removed_features)

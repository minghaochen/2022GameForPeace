import pandas as pd
import datatable as dt
from tqdm import tqdm

cols = ["worldid","iuin","dtgamestarttime","igameseq","isubgamemode","irankingame","champion_id","premade_size",
        "team","champion_used_exp","spell1","spell2","tier","queue"]
test = pd.read_csv('battle_data_one_day.txt', sep='|', names=cols)
# 选择测试数据里胡
test_uid = set(test['iuin'].unique().tolist())

#
cols = ["worldid", "iuin","dtgamestarttime","iduration","igameseq","isubgamemode","irankingame","idamgedealt",
        "idamagetaken","igoldearned","ihealingdone","ilargestkillingspree","ilargestmultikill","imagicdamagedealt",
        "imagicdamagetaken","iminionskilled","ineutralmonsterkills","iphysicaldamagedealt","iphysicaldamagetaken",
        "champion_id","premade_size","elo_change","team","champions_killed","num_deaths","assists","game_score",
        "flag","ext_flag","champion_used_exp","spell1","spell2","win","tier","queue"]

data = dt.fread("battle_data_one_month.txt", columns=cols)
# temp = data[:1000,:]


train_uid = set(dt.unique(data['iuin']).to_list()[0])
print(1)



# # 剔除不在测试机里
# rows = data.shape[0]
# select_bool = []
# for i in tqdm(range(rows)):
#         if data[i,'iuin'] in test_uid:
#                 select_bool.append(True)
#         else:
#                 select_bool.append(False)
# data = data[select_bool, :]


# # 筛选出3.17后的数据
from datetime import datetime
# time = datetime(2020, 3, 11, 0, 0)
# rows = data.shape[0]
# select_bool = []
# for i in tqdm(range(rows)):
#         if data[i,'dtgamestarttime'] > time:
#                 select_bool.append(True)
#         else:
#                 select_bool.append(False)
# data = data[select_bool, :]


# 筛选出大乱斗
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if data[i,'isubgamemode'] == 20:
                select_bool.append(True)
        else:
                select_bool.append(False)
data = data[select_bool, :]


# 剔除重新开局
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if data[i,'elo_change'] != 0:
                select_bool.append(True)
        else:
                select_bool.append(False)
data = data[select_bool, :]




# 特征数据集
time1 = datetime(2020, 3, 4, 0, 0)
time2 = datetime(2020, 3, 18, 0, 0)
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if (data[i,'dtgamestarttime'] > time1) and (data[i,'dtgamestarttime'] < time2):
                select_bool.append(True)
        else:
                select_bool.append(False)
feature_data = data[select_bool, :]
feature_data.to_csv('feature_data.csv')
# 训练数据集
time1 = datetime(2020, 3, 18, 0, 0)
time2 = datetime(2020, 3, 25, 0, 0)
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if (data[i,'dtgamestarttime'] > time1) and (data[i,'dtgamestarttime'] < time2):
                select_bool.append(True)
        else:
                select_bool.append(False)
train_data = data[select_bool, :]
train_data.to_csv('train_data.csv')
# 验证数据集
time1 = datetime(2020, 3, 25, 0, 0)
time2 = datetime(2020, 3, 26, 0, 0)
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if (data[i,'dtgamestarttime'] > time1) and (data[i,'dtgamestarttime'] < time2):
                select_bool.append(True)
        else:
                select_bool.append(False)
valid_data = data[select_bool, :]
valid_data.to_csv('valid_data.csv')

# 测试集特征数据集
time1 = datetime(2020, 3, 18, 0, 0)
time2 = datetime(2020, 4, 1, 0, 0)
rows = data.shape[0]
select_bool = []
for i in tqdm(range(rows)):
        if (data[i,'dtgamestarttime'] > time1) and (data[i,'dtgamestarttime'] < time2):
                select_bool.append(True)
        else:
                select_bool.append(False)
test_fea_data = data[select_bool, :]
test_fea_data.to_csv('test_fea_data.csv')



print(data.shape)
print(feature_data.shape)
print(train_data.shape)
print(valid_data.shape)
print(feature_data.shape[0]+train_data.shape[0]+valid_data.shape[0])

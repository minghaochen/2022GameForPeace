import pandas as pd
from tqdm import tqdm

train_feature = True
if train_feature:
    feature_data = pd.read_csv('feature_data.csv')
# feature_data = pd.read_csv('train_data.csv')
# feature_data = pd.read_csv('valid_data.csv')
else:
    feature_data = pd.read_csv('test_fea_data.csv')

    def int_map(x):
        try:
            gender = x if type(x) == int else eval(x)
        except:
            gender = 0
        return gender

    clean_col = ['iphysicaldamagetaken','champion_id','premade_size','elo_change','spell1','spell2']
    for col in clean_col:
        feature_data[col] = feature_data[col].map(int_map)
        feature_data = feature_data.drop(feature_data[feature_data[col] == 0].index)


feature_data = feature_data.drop(columns=['dtgamestarttime', 'iduration','igameseq', 'isubgamemode'])


uid = {'iuin': list(feature_data['iuin'].unique()),
       'worldid': [1 for i in range(len(list(feature_data['iuin'].unique())))]}
user_info = pd.DataFrame(uid)


user_info = user_info.join(feature_data.groupby('iuin').agg({'win':'mean'}), on='iuin').rename(columns={"win": "win_mean"})
user_info = user_info.join(feature_data.groupby('iuin').agg({'win':'sum'}), on='iuin').rename(columns={"win": "win_sum"})
user_info = user_info.join(feature_data.groupby('iuin').agg({'win':'count'}), on='iuin').rename(columns={"win": "win_count"})

user_info = user_info.join(feature_data.groupby('iuin').agg({'irankingame':'mean'}), on='iuin').rename(columns={"irankingame": "irankingame_mean"})
user_info = user_info.join(feature_data.groupby('iuin').agg({'irankingame':'max'}), on='iuin').rename(columns={"irankingame": "irankingame_max"})
user_info = user_info.join(feature_data.groupby('iuin').agg({'irankingame':'min'}), on='iuin').rename(columns={"irankingame": "irankingame_min"})
user_info['irankingame_delta'] = user_info["irankingame_max"] - user_info["irankingame_min"]


make_feature = ["idamgedealt","idamagetaken","igoldearned","ihealingdone","ilargestkillingspree","ilargestmultikill",
                "imagicdamagedealt","imagicdamagetaken","iminionskilled","ineutralmonsterkills","iphysicaldamagedealt",
                "iphysicaldamagetaken", "premade_size", "elo_change", "champions_killed", "num_deaths","assists",
                "game_score"]

for fea in make_feature:
    user_info = user_info.join(feature_data.groupby('iuin').agg({f'{fea}': 'mean'}), on='iuin').rename(
        columns={f"{fea}": f"{fea}_mean"})
    user_info = user_info.join(feature_data.groupby('iuin').agg({f'{fea}': 'std'}).fillna(0), on='iuin').rename(
        columns={f'{fea}': f"{fea}_std"})


# 还要考虑其他特征
def flag_map(x):
    x = hex(x)
    if x in ['0x1','0x2','0x4','0x01','0x02','0x04']:
        return 1
    else:
        return 0
feature_data['flag'] = feature_data['flag'].map(flag_map)
def ext_flag_map(x):
    x = hex(x)
    if x in ['0x1','0x800','0x1000','0x2000','0x8000','0x10000','0x20000','0x40000',
             '0x80000','0x100000','0x200000','0x800000']:
        return 1
    else:
        return 0
feature_data['ext_flag'] = feature_data['ext_flag'].map(ext_flag_map)

make_feature = ['flag','ext_flag']
for fea in make_feature:
    user_info = user_info.join(feature_data.groupby('iuin').agg({f'{fea}': 'mean'}), on='iuin').rename(
        columns={f"{fea}": f"{fea}_mean"})
    user_info = user_info.join(feature_data.groupby('iuin').agg({f'{fea}': 'sum'}), on='iuin').rename(
        columns={f"{fea}": f"{fea}_sum"})


print(1)

user_info.to_csv('train_user_feature.csv',index=None)
# user_info.to_csv('test_user_feature.csv',index=None)
print(1)


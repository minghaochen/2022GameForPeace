import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

feature_data = pd.read_csv('train_data.csv')
feature_data = feature_data[['iuin','igameseq','irankingame','champion_id',
                             'premade_size','team','champion_used_exp',
                             'spell1','spell2','tier','queue','win']].copy()
valid_data = pd.read_csv('valid_data.csv')
valid_data = valid_data[['iuin','igameseq','irankingame','champion_id',
                             'premade_size','team','champion_used_exp',
                             'spell1','spell2','tier','queue','win']].copy()

def int_map(x):
    try:
        gender = x if type(x) == int else eval(x)
    except:
        gender = 0
    return gender

clean_col = ['champion_id','spell1','spell2']
for col in clean_col:
    feature_data[col] = feature_data[col].map(int_map)
    feature_data = feature_data.drop(feature_data[feature_data[col] == 0].index)
    valid_data[col] = valid_data[col].map(int_map)
    valid_data = valid_data.drop(valid_data[valid_data[col] == 0].index)


feature = pd.read_csv('train_user_feature.csv')

feature_data = pd.merge(feature_data,feature,how='left',on='iuin').fillna(0)
valid_data = pd.merge(valid_data,feature,how='left',on='iuin').fillna(0)
print(1)

# 剔除不完整的游戏
df = feature_data.groupby('igameseq')
imcompleted = []
for keys in tqdm(df.groups):
    if len(df.groups[keys]) != 10:
        imcompleted.append(keys)
feature_data = feature_data.loc[~feature_data['igameseq'].isin(imcompleted)]

df = valid_data.groupby('igameseq')
imcompleted = []
for keys in tqdm(df.groups):
    if len(df.groups[keys]) != 10:
        imcompleted.append(keys)
valid_data = valid_data.loc[~valid_data['igameseq'].isin(imcompleted)]



# 按igameseq 和 team 排序
feature_data = feature_data.sort_values(by=['igameseq','team'], ascending=[True, True])
train_y = feature_data['win'].values
train_y = train_y.reshape(int(train_y.shape[0]/10),-1)
y_sum = np.sum(train_y,axis=1)
y_select = y_sum == 5
train_y = train_y[y_select][:,0]

# valid
valid_data = valid_data.sort_values(by=['igameseq','team'], ascending=[True, True])
valid_y = valid_data['win'].values
valid_y = valid_y.reshape(int(valid_y.shape[0]/10),-1)
y_sum = np.sum(valid_y,axis=1)
y_select_valid = y_sum == 5
valid_y = valid_y[y_select_valid][:,0]


train_spell = feature_data[['spell1', 'spell2']].copy()
valid_spell = valid_data[['spell1', 'spell2']].copy()

feature_data = feature_data[['irankingame', 'premade_size', 'champion_used_exp',
                             # 'spell1', 'spell2', 'tier', 'queue',
                             'win_mean', 'win_sum', 'win_count', 'irankingame_mean',
                             'irankingame_max', 'irankingame_min', 'irankingame_delta',
                             'idamgedealt_mean', 'idamgedealt_std', 'idamagetaken_mean',
                             'idamagetaken_std', 'igoldearned_mean', 'igoldearned_std',
                             'ihealingdone_mean', 'ihealingdone_std', 'ilargestkillingspree_mean',
                             'ilargestkillingspree_std', 'ilargestmultikill_mean',
                             'ilargestmultikill_std', 'imagicdamagedealt_mean',
                             'imagicdamagedealt_std', 'imagicdamagetaken_mean',
                             'imagicdamagetaken_std', 'iminionskilled_mean', 'iminionskilled_std',
                             'ineutralmonsterkills_mean', 'ineutralmonsterkills_std',
                             'iphysicaldamagedealt_mean', 'iphysicaldamagedealt_std',
                             'iphysicaldamagetaken_mean', 'iphysicaldamagetaken_std',
                             'premade_size_mean', 'premade_size_std', 'elo_change_mean',
                             'elo_change_std', 'champions_killed_mean', 'champions_killed_std',
                             'num_deaths_mean', 'num_deaths_std', 'assists_mean', 'assists_std',
                             'game_score_mean', 'game_score_std','flag_mean', 'flag_sum',
                             'ext_flag_mean', 'ext_flag_sum']]
valid_data = valid_data[['irankingame', 'premade_size', 'champion_used_exp',
                             # 'spell1', 'spell2', 'tier', 'queue',
                             'win_mean', 'win_sum', 'win_count', 'irankingame_mean',
                             'irankingame_max', 'irankingame_min', 'irankingame_delta',
                             'idamgedealt_mean', 'idamgedealt_std', 'idamagetaken_mean',
                             'idamagetaken_std', 'igoldearned_mean', 'igoldearned_std',
                             'ihealingdone_mean', 'ihealingdone_std', 'ilargestkillingspree_mean',
                             'ilargestkillingspree_std', 'ilargestmultikill_mean',
                             'ilargestmultikill_std', 'imagicdamagedealt_mean',
                             'imagicdamagedealt_std', 'imagicdamagetaken_mean',
                             'imagicdamagetaken_std', 'iminionskilled_mean', 'iminionskilled_std',
                             'ineutralmonsterkills_mean', 'ineutralmonsterkills_std',
                             'iphysicaldamagedealt_mean', 'iphysicaldamagedealt_std',
                             'iphysicaldamagetaken_mean', 'iphysicaldamagetaken_std',
                             'premade_size_mean', 'premade_size_std', 'elo_change_mean',
                             'elo_change_std', 'champions_killed_mean', 'champions_killed_std',
                             'num_deaths_mean', 'num_deaths_std', 'assists_mean', 'assists_std',
                             'game_score_mean', 'game_score_std','flag_mean', 'flag_sum','ext_flag_mean', 'ext_flag_sum']]


data = feature_data.values
valid_data = valid_data.values
train_spell = train_spell.values
valid_spell = valid_spell.values
# standized

train_x = data.reshape(int(data.shape[0]/10), int(data.shape[1]*10))
train_x = train_x[y_select]

valid_x = valid_data.reshape(int(valid_data.shape[0]/10), int(valid_data.shape[1]*10))
valid_x = valid_x[y_select_valid]

train_spell = train_spell.reshape(int(train_spell.shape[0]/10), int(train_spell.shape[1]*10))
valid_spell = valid_spell.reshape(int(valid_spell.shape[0]/10), int(valid_spell.shape[1]*10))
train_spell = train_spell[y_select]
valid_spell = valid_spell[y_select_valid]

from torch.utils.data import Dataset
class LOLDataset(Dataset):
    def __init__(self, data, label, spell):
        self.data = data
        self.label = label
        self.spell = spell

    def __getitem__(self, index):

        blue = self.data[index,0:50*5].reshape(5, 50)
        red = self.data[index,50*5:].reshape(5, 50)
        label = self.label[index]
        spell = self.spell[index]

        return  blue, red, label, spell

    def __len__(self):
        return self.data.shape[0]

class Victory(nn.Module):
    def __init__(self):
        super(Victory, self).__init__()

        # self.attention1 = nn.MultiheadAttention(46, 2)
        # self.attention2 = nn.MultiheadAttention(46, 2)

        self.attention1 = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=50+32, nhead=2),
            nn.TransformerEncoderLayer(d_model=50+32, nhead=2)
        )

        self.attention2 = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=50+32, nhead=2),
            nn.TransformerEncoderLayer(d_model=50+32, nhead=2)
        )
        self.MLP = nn.Linear(82*3, 2)

        self.spell_emb = nn.Embedding(50, 16)

        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)


    def forward(self, blue, red, spell, criterion, targets=None):
        blue_spell = spell[:,0:10].reshape(-1,5,2)
        red_spell = spell[:,10:].reshape(-1,5,2)
        blue_spell = self.spell_emb(blue_spell).reshape(-1,5,32)
        red_spell = self.spell_emb(red_spell).reshape(-1,5,32)
        blue = torch.cat((blue,blue_spell),2)
        red = torch.cat((red,red_spell),2)
        blue = torch.transpose(blue, 0, 1)
        red = torch.transpose(red, 0, 1)
        # blue, blue_weights = self.attention1(blue, blue, blue)
        # red, red_weights = self.attention1(red, red, red)
        blue = self.attention1(blue)
        red = self.attention1(red)
        diff = blue - red
        # diff, _ = self.attention2(diff, diff, diff)
        diff = self.attention2(diff)
        blue = torch.transpose(blue, 0, 1)
        red = torch.transpose(red, 0, 1)
        diff = torch.transpose(diff, 0, 1)

        total = torch.cat((torch.mean(diff,1),torch.mean(blue,1),torch.mean(red,1)),1)

        total = self.dropout(total)

        logits1 = self.MLP(self.dropout1(total))
        logits2 = self.MLP(self.dropout2(total))
        logits3 = self.MLP(self.dropout3(total))
        logits4 = self.MLP(self.dropout4(total))
        logits5 = self.MLP(self.dropout5(total))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        logits = torch.softmax(logits, dim=-1)

        loss = 0
        if targets is not None:
            loss1 = criterion(logits1, targets)
            loss2 = criterion(logits2, targets)
            loss3 = criterion(logits3, targets)
            loss4 = criterion(logits4, targets)
            loss5 = criterion(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            return logits, loss

        # logit = self.MLP(total)
        return logits, loss
def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for blue, red, label, spell in tqdm(train_loader):

        blue = blue.to(device).float()
        red = red.to(device).float()
        label = label.to(device)
        spell = spell.to(device).long()

        pred, loss = model(blue, red, spell, criterion, label)

        # loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        model.zero_grad()


        #
        _, indices = torch.max(pred, dim=1)
        prediction = indices.long().cpu().numpy()

        pred_list.extend(prediction)
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = accuracy_score(np.array(label_list), np.array(pred_list))

    return score


def validate(model, val_loader, device, criterion):
    model.eval()

    pred_list = []
    label_list = []

    for blue, red, label,spell in tqdm(val_loader):

        blue = blue.to(device).float()
        red = red.to(device).float()
        label = label.to(device)
        spell = spell.to(device).long()
        # pred = model(blue, red, spell)
        pred, loss = model(blue, red, spell, criterion, label)


        _, indices = torch.max(pred, dim=1)
        prediction = indices.long().cpu().numpy()

        pred_list.extend(prediction)

        # pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(label.squeeze().cpu().detach().numpy())

    score = accuracy_score(np.array(label_list), np.array(pred_list))

    return score

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10,
                      shuffle=True,
                      random_state=2022)



model = Victory().cuda()
print(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = 'cuda'

k_score = []

from sklearn.preprocessing import MinMaxScaler
import joblib
import copy
import pickle
mm = MinMaxScaler()
X_train = mm.fit_transform(train_x)

pickle.dump(mm, open("MinMaxScaler.pickle", "wb"))
mm = pickle.load(open("MinMaxScaler.pickle", "rb"))

X_test = mm.transform(valid_x)



train_dataset = LOLDataset(X_train, train_y, train_spell)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)

test_dataset = LOLDataset(X_test, valid_y, valid_spell)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

best_score = 0
early_stop = 0
for epoch in range(100):
    train_score = fit(model, train_loader, optimizer, criterion, device)
    val_score = validate(model, test_loader, device, criterion)

    if epoch % 5 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.9

    if val_score > best_score:
        best_model = copy.deepcopy(model)
        best_score = val_score
        print(
            f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} '
        )
        model_name = f'best_model'
        torch.save(best_model.state_dict(), model_name + '.pt')
        early_stop = 0
    else:
        early_stop += 1

    if early_stop >= 10:
        break




print('=====================k flod score======================')
print(best_score)
print('=======================================================')
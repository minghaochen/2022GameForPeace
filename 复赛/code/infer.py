import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


cols = ["worldid","iuin","dtgamestarttime","igameseq","isubgamemode","irankingame","champion_id","premade_size",
        "team","champion_used_exp","spell1","spell2","tier","queue"]
test = pd.read_csv('battle_data_one_day.txt', sep='|', names=cols)
# 选择测试数据里胡
test_uid = set(test['iuin'].unique().tolist())


feature = pd.read_csv('test_user_feature.csv')

test = pd.merge(test,feature,how='left',on='iuin').fillna(0)

# 按igameseq 和 team 排序
test = test.sort_values(by=['igameseq','team'], ascending=[True, True])

teamid = test['igameseq'].values
teamid = teamid[::10]
res = pd.DataFrame(teamid,columns=['gameid'])
# res['result（1胜，0负）'] = 0

test_spell = test[['spell1', 'spell2']].copy()

test = test[['irankingame', 'premade_size', 'champion_used_exp',
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


data = test.values
test_spell = test_spell.values

data = data.reshape(int(data.shape[0]/10), int(data.shape[1]*10))
test_spell = test_spell.reshape(int(test_spell.shape[0]/10), int(test_spell.shape[1]*10))



from torch.utils.data import Dataset
class LOLDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):

        blue = self.data[index,0:46*5].reshape(5, 46)
        red = self.data[index,46*5:].reshape(5, 46)
        label = self.label[index]

        return  blue, red, label

    def __len__(self):
        return self.data.shape[0]

class LOLDataset_test(Dataset):
    def __init__(self, data, spell):
        self.data = data
        self.spell = spell

    def __getitem__(self, index):
        blue = self.data[index, 0:50 * 5].reshape(5, 50)
        red = self.data[index, 50 * 5:].reshape(5, 50)
        spell = self.spell[index]

        return  blue, red, spell

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

model = Victory().cuda()
model.load_state_dict(torch.load('best_model.pt'))
from sklearn.preprocessing import MinMaxScaler
#导入训练集
import pickle
# mm = MinMaxScaler()
mm = pickle.load(open("MinMaxScaler.pickle", "rb"))
X_test = mm.transform(data)

test_dataset = LOLDataset_test(X_test,test_spell)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0)

model.eval()
criterion = nn.CrossEntropyLoss()
pred_list = []
label_list = []
device = 'cuda'
for blue, red, spell in tqdm(test_loader):

    blue = blue.to(device).float()
    red = red.to(device).float()
    spell = spell.to(device).long()
    pred, loss = model(blue, red, spell, criterion)

    #
    _, indices = torch.max(pred, dim=1)
    prediction = indices.long().cpu().numpy()
    pred_list.extend(prediction)


# res['teamid'] = 100
res['result(1胜，0负)'] = np.array(pred_list)

result = pd.read_csv('result.csv',encoding='gbk')
result = result.drop(columns='result(1胜，0负)')

result = pd.merge(result, res, on='gameid')


result.to_csv('submisson.csv',index=None, encoding='gbk')
print(1)
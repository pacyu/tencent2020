import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import lightgbm as lgb
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ti import session


ti_session = session.Session()

ad = pd.read_csv('train_preliminary/ad.csv')
cl = pd.read_csv('train_preliminary/click_log.csv')
ud = pd.read_csv('train_preliminary/user.csv')

ad_test = pd.read_csv('test/ad.csv')
cl_test = pd.read_csv('test/click_log.csv')

ad['product_id'][ad['product_id'] == '\\N'] = 0
ad_test['product_id'][ad_test['product_id'] == '\\N'] = 0

ad['industry'][ad['industry'] == '\\N'] = 0
ad_test['industry'][ad_test['industry'] == '\\N'] = 0

new_df = pd.merge(cl, ad, on='creative_id', how='left')
new_df_test = pd.merge(cl_test, ad_test, on='creative_id', how='left')
new_df = new_df.astype(np.int)
new_df_test = new_df_test.astype(np.int)

# print(new_df)
# print('='*80)
# print(new_df_test)

train_x = new_df.groupby([new_df['user_id']]).mean().values
test_x = new_df_test.groupby([new_df_test['user_id']]).mean().values
y_age = ud['age']
y_gender = ud['gender']

X_train, X_test, y_train_age, y_test_age = train_test_split(train_x, y_age, test_size=0.4, random_state=2020)
X_train_, X_test_, y_train_gender, y_test_gender = train_test_split(train_x, y_gender, test_size=0.4, random_state=2020)

clf_age = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, objective='multiclass',
    max_depth=-1, n_estimators=1500, n_classes=10,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2020, n_jobs=-1
)
clf_gender = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, objective='binary',
    max_depth=-1, n_estimators=1500, n_classes=2,
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2020, n_jobs=-1
)

clf_age.fit(X_train, y_train_age)
y_lgb_test_pred_age = clf_age.predict(X_test)
clf_gender.fit(X_train_, y_train_gender)
y_lgb_test_pred_gender = clf_gender.predict(X_test_)

print(y_test_age)
print('='*50)
print(y_lgb_test_pred_age)
test_lgb_acc_age = accuracy_score(y_test_age, y_lgb_test_pred_age)
print('age accuracy:', test_lgb_acc_age)
print('='*80)
print(y_test_gender)
print('='*50)
print(y_lgb_test_pred_gender)
test_lgb_acc_gender = accuracy_score(y_test_gender, y_lgb_test_pred_gender)
print('gender accuracy:', test_lgb_acc_gender)

# uides = list(cl_test['user_id'].groupby([cl_test['user_id']]).last())
reage = True
regender = True
if regender and reage:  # 重建数据
    submission = pd.read_csv('result/submission.csv')
    submission['predicted_age'] = y_lgb_test_pred_age.reshape(len(y_lgb_test_pred_age))
    submission['predicted_gender']: y_lgb_test_pred_gender.reshape(len(y_lgb_test_pred_gender))
    sdf = pd.DataFrame(submission)
    sdf.to_csv('result/submission.csv', index=False)
elif regender:  # 如果仅修改性别
    submission = pd.read_csv('result/submission.csv')
    submission['predicted_gender'] = y_lgb_test_pred_gender.reshape(len(y_lgb_test_pred_gender))
    submission.to_csv('result/submission.csv', index=False)
elif reage:  # 如果仅修改年龄
    submission = pd.read_csv('result/submission.csv')
    submission['predicted_age'] = y_lgb_test_pred_age.reshape(len(y_lgb_test_pred_age))
    submission.to_csv('result/submission.csv', index=False)
ti_session = session.Session()
inputs = ti_session.upload_data(path='result', bucket="2020cos-1301738573", key_prefix="contest")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# class Model(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, n_classes):
#         super(Model, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, vocab_size)
#         self.fc = nn.Linear(embedding_dim, n_classes)
#
#     def forword(self, input):
#         x = self.embedding(input).view((-1, 1))
#         x = self.lstm(x)
#         return self.fc(x)
#
#
# ad = pd.read_csv('E:/Tencent2020/train_preliminary/ad.csv')
# cl = pd.read_csv('E:/Tencent2020/train_preliminary/click_log.csv')
# ur = pd.read_csv('E:/Tencent2020/train_preliminary/user.csv')
# ad['product_id'][ad['product_id'] == '\\N'] = 0
# ad['industry'][ad['industry'] == '\\N'] = 0
# data = pd.merge(cl, ad, how='left', on='creative_id')
# print(cl)
# print(cl['creative_id'].groupby([cl['user_id']]).groups)
# train_x = [list(data['creative_id'].groupby([data['user_id']])), ur['age'].values]
# dataLoader = DataLoader(train_x, len(train_x) // 1000, shuffle=True, num_workers=2)
# print(train_x)
# # vocab = {word: i for i, word in enumerate(data['creative_id'])}
# vocab_size = len(train_x)
# loss_function = nn.NLLLoss()
# age_model = Model(vocab_size, 128, 10).to(device)
# optimizer = optim.SGD(age_model.parameters(), lr=0.001)
# for epoch in range(500):
#     for i, batch_x in enumerate(dataLoader):
#         age_model.zero_grad()
#         y_age_pred = age_model(batch_x)
#         loss = loss_function(y_age_pred, ur['age'])
#         loss.backward()
#         optimizer.step()
#         print('epoch:', epoch, 'step:', i, 'loss:', loss)
#         if epoch % 5:
#             torch.save(age_model.state_dict(), 'age_model_epoch_%d.pth' % epoch)  # 保存训练模型

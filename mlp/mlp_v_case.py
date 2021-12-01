import numpy as np
import pandas as pd

#  we use chunksize to avoid memeory problems
from matplotlib import pyplot as plt


df_death = pd.read_csv("data_table_for_daily_case_trends__the_united_states.csv", header=2, usecols=["New Cases", "Date"])
df_death_new = df_death["New Cases"][:322][::-1]
result = np.array(df_death_new)
print(df_death_new)

df = pd.read_csv('vaccinations_in_the_us.csv', header=2, usecols=['Date Type', 'Date', 'Daily Count People Receiving Dose 1', 'People Receiving 1 or More Doses Cumulative'])
df = df[df['Date Type'] == "Admin"]


sentiment140 = df[['Daily Count People Receiving Dose 1', 'People Receiving 1 or More Doses Cumulative']]
print(sentiment140)

data = np.array(df)
#result = data[:, 2][:100]
result_for_pre = data[:, 2][100:200]
sentimental = pd.read_csv('SentimentIndex.csv', header=0)
sentimental.dropna(inplace=True)
#sentiment140 = sentimental[['sentiment140', 'textblob2']][12:112]
sent140_for_pre = sentimental[['sentiment140', 'textblob2']][112:212]
textblob = sentimental['textblob2'][12:112]
vader = sentimental['vader']
n = sentimental['n']

# read file second way
# with open and for line in f
# import csv
# with open('vaccinations_in_the_us.csv', 'r') as f:
#   reader = csv.reader(f)
#   data = [row for row in reader]
# print(data)
# f is iterable object
# for line in f:
#   # will use buffered IO memory management
#   print(line)


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

# 导入数据
housedata = fetch_california_housing()
# 划分测试集和训练集
housedata.data = sentiment140
housedata.target = result
housedata.feature_names = ['sentiment140', 'textblob2']
X_train, X_test, y_train, y_test = train_test_split(housedata.data, housedata.target, test_size=0.3, random_state=42)

# 标准化处理
scale = StandardScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.fit_transform(X_test)
housedatadf = pd.DataFrame(data=X_train_s, columns=housedata.feature_names)
housedatadf["target"] = y_train
housedatadf.dropna(inplace=True)

# #可视化训练数据的相关系数热力图 2个自变量以及目标之间的相关系数
# datacor=np.corrcoef(housedatadf.values,rowvar=0)
# datacor=pd.DataFrame(data=datacor,columns=housedatadf.columns,index=housedatadf.columns)
# plt.figure(figsize=(8,6))
# ax=sns.heatmap(datacor,square=True,annot=True,fmt=".3f",linewidths=5,cmap="YlGnBu",cbar_kws={"fraction":0.046,"pad":0.03})
# plt.show()

# 将数据集转化为张量 并处理为PyTorch网络使用的数据
print(X_train_s, y_train, X_test_s, y_test)
train_xt = torch.from_numpy(X_train_s.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test_s.astype(np.float32))
test_yt = torch.from_numpy(y_test.astype(np.float32))
# 将数据处理为数据加载器
train_data = Data.TensorDataset(train_xt, train_yt)
test_data = Data.TensorDataset(test_xt, test_yt)
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)


# 搭建MLP回归模型
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(in_features=2, out_features=100, bias=True)  # 2*100 2个属性特征
        # 定义第二个隐藏层
        self.hidden2 = nn.Linear(100, 100)  # 100*100
        # 定义第三个隐藏层
        self.hidden3 = nn.Linear(100, 50)  # 100*50
        # 回归预测层
        self.predict = nn.Linear(50, 1)  # 50*1  预测只有一个 房价

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # print(output)
        return output[:, 0]


mlpreg = MLPregression()
# print(mlpreg)

# 定义优化器
optimizer = torch.optim.Adam(mlpreg.parameters(), lr=0.01)
loss_func = nn.MSELoss()
train_loss_all = []
for epoch in range(1000):
    train_loss = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlpreg(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
        # print()
    train_loss_all.append(train_loss / train_num)
    # print(train_loss_all[-1])
plt.figure(figsize=(10, 6))
plt.plot(train_loss_all, "ro-", label="Train loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 预测
pre_y = mlpreg(test_xt)
pre_y = pre_y.data.numpy()
print(y_test, pre_y, "y and pre y ")
mae = mean_absolute_error(y_test, pre_y)

index = np.argsort(y_test)
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(y_test)), y_test[index], "r", label="original y")
plt.scatter(np.arange(len(pre_y)), pre_y[index], s=3, c="b", label="prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("index")
plt.ylabel("y")
plt.show()

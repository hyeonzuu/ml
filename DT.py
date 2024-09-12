import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.head()
df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                   'bill_depth_mm': 'x'})
df
# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
29.81
# x=15 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹
# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]
# 각 그룹 MSE는 얼마 인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)
# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1* n1 + mse2 * n2)/(n1+n2)
29.23
29.81 - 29.23
# x = 20일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0]  # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)
(mse1* n1 + mse2 * n2)/(n1+n2)
29.73

29.81-29.73

# df=df.query("x >= 16.41")

# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x):
    n1=df.query(f"x < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)
    return float((mse1* n1 + mse2 * n2)/(n1+n2))

my_mse(20)
my_mse(16.5)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(13.2, 21.4, 0.01)
result=np.repeat(0.0, 820)
for i in range(820):
    x_values=np.arange(16.51, 21.5, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)]
16.41

# 14.01, 16.42, 19.4
# -------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

x_values1 = np.arange(group1['x'].min()+0.01, group1['x'].max(), 0.01)
result1 = np.repeat(0.0, len(x_values1))
for i in range(0, len(x_values1)):
    result1[i] = my_mse(group1, x_values1[i])
x_values1[np.argmin(result1)] # 14.01
result1.min()

x_values2 = np.arange(group2['x'].min() + 0.01, group2['x'].max(), 0.01)
result2 = np.repeat(0.0, len(x_values2))
for i in range(0, len(x_values2)):
    result2[i] = my_mse(group2, x_values2[i])
x_values[np.argmin(result2)] # 19.4

# x, y 산점도 그래프 & 평행선 4개
thresholds = [14.01, 16.42, 19.4]
df['group'] = np.digitize(df['x'], thresholds)
y_mean = df.groupby('group').mean()['y']

k1 = np.linspace(13, 14.01, 100)
k2 = np.linspace(14.01, 16.42, 100)
k3 = np.linspace(16.42, 19.4, 100)
k4 = np.linspace(19.4, 22, 100)

df.plot(kind='scatter', x='x', y='y', color = 'black', s=5)
plt.axvline(x=16.4, color='b', linestyle=':')
plt.axvline(x=14.01, color='r', linestyle=':')
plt.axvline(x=19.4, color='r', linestyle=':')

plt.scatter(k1, np.repeat(y_mean[0], 100), color='r', s=5)
plt.scatter(k2, np.repeat(y_mean[1], 100), color='r', s=5)
plt.scatter(k3, np.repeat(y_mean[2], 100), color='r', s=5)
plt.scatter(k4, np.repeat(y_mean[3], 100), color='r', s=5)
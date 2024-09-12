import pandas as pd
import numpy as np

admission_data = pd.read_csv("./data/admission.csv")
print(admission_data.shape)


# GPA: 학점
# GRE: 대학원 입학시험 (영어, 수학)
# 합격을 한 사건: Admit
# Admit의 확률 오즈(Odds)는?
# P(Admit)=합격 인원 / 전체 학생
p_hat = admission_data['admit'].mean()
p_hat / (1 - p_hat)
# p(A): 0.5보다 큰 경우 -> 오즈비: 무한대에 가까워짐
# p(A): 0.5 -> 오즈비: 1
# p(A): 0.5보다 작은 경우 -> 오즈비: 0에 가까워짐
# 확률의 오즈비가 갖는 값의 범위: 0~무한대
admission_data['rank'].unique()
grouped_data = admission_data \
    .groupby('rank', as_index=False) \
    .agg(p_admit=('admit', 'mean'), )
grouped_data['odds'] = grouped_data['p_admit'] / (1 - grouped_data['p_admit'])
print(grouped_data)
# 확률이 오즈비가 3!
# P(A)?
# admission 데이터 산점도 그리기
# x: gre, y: admit
# admission_data
import seaborn as sns
sns.stripplot(data=admission_data,
              x='rank', y='admit', jitter=0.3, alpha=0.3)
sns.scatterplot(data=grouped_data,
              x='rank', y='p_admit')
sns.regplot(data=grouped_data, x='rank', y='p_admit')
odds_data = admission_data.groupby('rank').agg(p_admit=('admit', 'mean')).reset_index()
odds_data['odds'] = odds_data['p_admit'] / (1 - odds_data['p_admit'])
odds_data['log_odds'] = np.log(odds_data['odds'])
print(odds_data)


sns.regplot(data=odds_data, x='rank', y='log_odds')



import statsmodels.api as sm
#!pip install statsmodels
model = sm.formula.ols("log_odds ~ rank", data=odds_data).fit()
print(model.summary())


np.exp(-0.5675)

import statsmodels.api as sm
admission_data = pd.read_csv("./data/admission.csv")

# admission_data['rank'] = admission_data['rank'].astype('category')
admission_data['gender'] = admission_data['gender'].astype('category')

model = sm.formula.logit("admit ~ gre + gpa + rank + gender", data=admission_data).fit()

print(model.summary())
입학할 확률의 오즈가 
np.exp(0.7753)

# 여학생
# GPA: 3.5
# GRE: 500
# Rank: 2

# 합격 확률 예측해보세요!
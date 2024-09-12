import pandas as pd
import statsmodels.api as sm
import numpy as np

# 문제 1. 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
df = pd.read_csv('./data/leukemia_remission.txt', delim_whitespace=True)
df

model = sm.formula.logit("REMISS ~ CELL + SMEAR + INFIL + LI + BLAST + TEMP", data=df).fit()
print(model.summary())

# 문제 2.
# 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
# 통계적으로 유의하다


# 문제 3.
# 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# P>|z|가 0.2보다 작은 LI, TEMP가 유의하다

# 문제 4. 다음 환자에 대한 오즈는 얼마인가요?
odds = np.exp(64.2581+ 0.65*30.8301+0.45*24.6863+0.55*-24.9745+1.2*4.3605+1.1*-0.0115+0.9*-100.1734)
odds # 0.03817459641135519

# 문제 5. 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
prob_remission = odds / (1 + odds)
prob_no_remission = 1 - prob_remission
print(f"백혈병 세포가 관측되지 않을 확률: {prob_no_remission}")

# 문제 6. TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
# TEMP 변수의 계수 확인
temp_coef = model.params['TEMP']
print(f"TEMP 변수의 계수: {temp_coef}")

# 문제 7. CELL 변수의 99% 오즈비에 대한 신뢰구간을 구하시오.
from scipy.stats import norm
# Beta__cell +- z0.005 *  52.135 
z005 = norm.ppf(0.995)  # 2.58
30.8301 - 2.58 * 52.135  # -103.678
30.8301 + 2.58 * 52.135  # 165.338
# 문제 8. 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
from sklearn.metrics import confusion_matrix

# 예측 확률 계산
pred_probs = model.predict(df)
# 50% 이상일 경우 1로 변환
predictions = (pred_probs >= 0.5).astype(int)

# 혼동 행렬 생성
cm = confusion_matrix(df['REMISS'], predictions)
print(f"혼동 행렬: \n{cm}")

# 문제 9. 해당 모델의 Accuracy는 얼마인가요?
# Accuracy 계산
accuracy = (predictions == df['REMISS']).mean()
print(f"Accuracy: {accuracy}")


# 문제 10. 해당 모델의 F1 Score를 구하세요.
from sklearn.metrics import f1_score

# F1 점수 계산
f1 = f1_score(df['REMISS'], predictions)
print(f"F1 Score: {f1}")
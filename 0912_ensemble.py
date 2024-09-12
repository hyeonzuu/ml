# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# 데이터 로드
house_train = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/ml/data/train.csv")
house_test = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/ml/data/test.csv")
sub_df = pd.read_csv("C:/Users/USER/Documents/LS빅데이터스쿨/ml/data/sample_submission.csv")

# 결측값 처리
df = pd.concat([house_train, house_test], ignore_index=True)

# 범주형 결측값 처리
qualitative = df.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    df[col].fillna("unknown", inplace=True)

# 숫자형 결측값 처리
quantitative = df.select_dtypes(include=[int, float])
fill_columns = quantitative.columns[quantitative.isna().sum() > 0]
train_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt']

# 결측값 선형 회귀 함수
def nan_regression(df, fill_column, train_columns):
    train_data = df.dropna(subset=[fill_column])
    X_train = train_data[train_columns].dropna()
    y_train = train_data.loc[X_train.index, fill_column]
    
    model_in = LinearRegression()
    model_in.fit(X_train, y_train)
    
    test_data = df[df[fill_column].isna()]
    X_test = test_data[train_columns].fillna(X_train.mean())
    predicted_values = model_in.predict(X_test)
    
    df.loc[df[fill_column].isna(), fill_column] = predicted_values

for fill_column in fill_columns:
    nan_regression(df, fill_column, train_columns)

#임의.
df['YearBuilt_GrLivArea'] = df['YearBuilt'] * df['GrLivArea']
df['TotalBsmt_1stFlrSF'] = df['TotalBsmtSF'] * df['1stFlrSF']
df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']


df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)
train_n = len(house_train)
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

train_x = train_df.drop(["SalePrice", "Id"], axis=1)
train_y = train_df["SalePrice"]
test_x = test_df.drop(["SalePrice", "Id"], axis=1)

# 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 첫 번째 층: ElasticNet, RandomForest, XGBoost

# ElasticNet
eln_model = ElasticNet()
param_grid_eln = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

grid_search_eln = GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid_eln,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_eln.fit(train_x_scaled, train_y)
best_eln_model = grid_search_eln.best_estimator_

# ElasticNet2
eln_model2 = ElasticNet()
param_grid_eln2 = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

grid_search_eln2 = GridSearchCV(
    estimator=eln_model2,
    param_grid=param_grid_eln2,
    scoring='neg_mean_squared_error',
    cv=3
)
grid_search_eln2.fit(train_x_scaled, train_y)
best_eln_model2 = grid_search_eln2.best_estimator_

# RandomForest
rf_model = RandomForestRegressor(n_estimators=100)
param_grid_rf = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_rf.fit(train_x_scaled, train_y)
best_rf_model = grid_search_rf.best_estimator_

# XGBoost
xgb_model = XGBRegressor()
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_xgb.fit(train_x_scaled, train_y)
best_xgb_model = grid_search_xgb.best_estimator_

# 첫 번째 층 모델 예측값
y1_hat = best_eln_model.predict(train_x_scaled)
y2_hat = best_eln_model2.predict(train_x_scaled)
y3_hat = best_rf_model.predict(train_x_scaled)
y4_hat = best_xgb_model.predict(train_x_scaled)


train_x_stack_1 = pd.DataFrame({
    'y1': y1_hat,
    'y2': y2_hat,
    'y3': y3_hat,
    'y4': y4_hat
})

# 두 번째 층: GradientBoosting, Ridge
# GradientBoostingRegressor 최적화(쓰라길래 써봄)
gb_model = GradientBoostingRegressor()
param_grid_gb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5]
}

grid_search_gb = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid_gb,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_gb.fit(train_x_stack_1, train_y)
best_gb_model = grid_search_gb.best_estimator_

# Ridge 최적화
rg_model = Ridge()
param_grid_rg = {
    'alpha': np.arange(0, 10, 0.01)
}

grid_search_rg = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid_rg,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_rg.fit(train_x_stack_1, train_y)
best_ridge_model = grid_search_rg.best_estimator_

# Lasso 최적화
lasso_model = Lasso()
param_grid_lasso = {
    'alpha': np.arange(0.01, 10, 0.1)
}

grid_search_lasso = GridSearchCV(
    estimator=lasso_model,
    param_grid=param_grid_lasso,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_lasso.fit(train_x_stack_1, train_y)
best_lasso_model = grid_search_lasso.best_estimator_


pred_y_eln = best_eln_model.predict(test_x_scaled)
pred_y_eln2 = best_eln_model.predict(test_x_scaled)
pred_y_rf = best_rf_model.predict(test_x_scaled)
pred_y_xgb = best_xgb_model.predict(test_x_scaled)

test_x_stack_1 = pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_eln2,
    'y3': pred_y_rf,
    'y4': pred_y_xgb

})

pred_y_gb = best_gb_model.predict(test_x_stack_1)
pred_y_ridge = best_ridge_model.predict(test_x_stack_1)
pred_y_lasso = best_lasso_model.predict(test_x_stack_1)


pred_y_gb_train = best_gb_model.predict(train_x_stack_1)
pred_y_ridge_train = best_ridge_model.predict(train_x_stack_1)
pred_y_lasso_train = best_lasso_model.predict(train_x_stack_1)

mse_gb = mean_squared_error(train_y, pred_y_gb_train)  
mse_ridge = mean_squared_error(train_y, pred_y_ridge_train) 
mse_lasso = mean_squared_error(train_y, pred_y_lasso_train)  

inv_mse_gb = 1 / mse_gb
inv_mse_ridge = 1 / mse_ridge
inv_mse_lasso = 1 / mse_lasso

total_inv_mse = inv_mse_gb + inv_mse_ridge + inv_mse_lasso
w_gb = inv_mse_gb / total_inv_mse
w_ridge = inv_mse_ridge / total_inv_mse
w_lasso = inv_mse_lasso / total_inv_mse

pred_y_final = (w_gb * pred_y_gb) + (w_ridge * pred_y_ridge) + (w_lasso * pred_y_lasso)

sub_df["SalePrice"] = pred_y_final

#sub_df.to_csv("./data/sample_submission_weighted_a`x`vg_mse.csv", index=False)

sub_df.head()  # 결과 확인


import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor

!gdown 1VzVeDOqWQMrdiN-QM32F1-Y3yB9-nv0c
!unzip mars-regr.zip -d dataset
!gdown 1J7pQpiXS-yJST2rbfRppEOjhl0combYC
!unzip /content/mars_final_private.zip

submit_sample = pd.read_csv("/content/dataset/mars-sample_submission-regr.csv")
submit_sample.head(50)
#посмотрим тестовые данных
#test = pd.read_csv("/content/mars-private_test-reg.csv")
#test.head(50)
#train
train = pd.read_csv("/content/dataset/mars-train-regr.csv")
train.head(50)
df = pd.DataFrame(test)
rows_with_missing_values = df[df.isnull().any(axis=1)]
print(rows_with_missing_values)
#все лейблы в диапозоне от 0 до 1
for column in train.columns[1:]:
  print(f"{column} : {train[column].min()}")
X_train = train.drop('Доля сигнала в ВП', axis=1)
y_train = train['Доля сигнала в ВП']
X_test = test

knn_model = KNeighborsRegressor(n_neighbors=15)
catboost_model = CatBoostRegressor(iterations= 2000, learning_rate=0.05, depth=7)

ens= VotingRegressor(estimators=[('knn', knn_model), ('catboost', catboost_model)])
ens.fit(X_train, y_train)
joblib.dump(ens, "ens_mod_reg.joblib")

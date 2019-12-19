#%%
import pandas as pd 
import numpy as np
import lightgbm as lgbm
import sklearn as sk
import xgboost as xgb
from sklearn import model_selection
from sklearn import ensemble
from sklearn import preprocessing as sk_prep

#%%

base_path = "./data/"
file = "churn_demo.csv"

df_base = pd.read_csv(base_path+file).drop(columns=['RowNumber', 'CustomerId'], axis = 1)

df_base.head(10)

#%%
print(df_base.shape)

#%%
#baseline random forest version 
label = 'Exited'
df_columns = df_base.columns[0 : df_base.shape[1]-1]

#%%
#encoders 
l_binarizer = sk_prep.LabelBinarizer()
str_encoder = sk_prep.OrdinalEncoder()

#binarize label
y_encoded = l_binarizer.fit_transform(y=df_base[label])

#encode strings
X_encoded = pd.DataFrame(str_encoder.fit_transform(X=df_base.drop(columns = [label], axis = 1)),
                            columns = df_columns)

X_encoded.head(10)
#%%
#split data in 70-30
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_encoded, y_encoded, test_size = 0.3)

print(X_train.shape, X_test.shape)

#%%
#100 trees of maximum 8 levels
rf = ensemble.RandomForestClassifier(n_estimators = 100, 
        max_depth=8, criterion="entropy", n_jobs=-1)
rf.fit(X = X_train, y = y_train.ravel())

#%%
from sklearn import metrics

preds_proba_rf = rf.predict_proba(X=X_test)
preds_rf = rf.predict(X=X_test)

#%%
print(preds_proba_rf)

#%%
print("Accuracy: ", round(metrics.accuracy_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Precision: ", round(metrics.precision_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Recall: ", round(metrics.recall_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("F1: ", round(metrics.f1_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))

#%%

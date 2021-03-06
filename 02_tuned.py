#%%
import pandas as pd 
import numpy as np
import lightgbm as lgbm
import sklearn as sk
import xgboost as xgb
from sklearn import model_selection
from sklearn import ensemble
from sklearn import preprocessing as sk_prep
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

#%%
#load data from our local data folder
base_path = "./data/"
file = "churn_demo.csv"

#drop features that will not be useful, they are either identifiers of the customer or 
#personal information without any relation with their churn likelihood
df_base = pd.read_csv(base_path+file).drop(columns=['RowNumber', 'Surname', 'CustomerId'], axis = 1)

df_base.head(10)

#%%
#how much data do we have?
print(df_base.shape)

#%%
#create metadata 
label = 'Exited'
df_columns = list(df_base.columns)
df_columns.remove(label)
print(df_columns)

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
#split data in 70-30 ratios
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_encoded, y_encoded, test_size = 0.3)

print(X_train.shape, X_test.shape)

#%%
#tune our random forest
param_grid = {
    "max_depth": list(range(20, 300, 10)),
    "criterion":  ["gini", "entropy"],
    "max_features":  ["auto", "sqrt"],
    "min_samples_leaf": list(range(1, 5, 1)),
    "min_samples_split": list(range(2, 20, 2)),
    "n_estimators": [int(elems) for elems in list(np.logspace(np.log(100), np.log(800), base = np.exp(1), num = 30))]
}

#use 4 cores for each classifier and 2 models in parallel
rf = ensemble.RandomForestClassifier(n_jobs = 4)
n_iters = 20

random_search = model_selection.RandomizedSearchCV(estimator = rf, 
                                                   param_distributions = param_grid, 
                                                   n_iter = n_iters, scoring = 'f1', cv = 5, 
                                                   verbose = 1, n_jobs = 2)

#%%
#search in the random space
random_search.fit(X=X_train, y=y_train.ravel())

#%%
#get best model
best_rf = random_search.best_estimator_ 

#show best parameters
best_params_rf = random_search.best_params_
best_params_rf

#%%
#predict using our test set and print different error measures
from sklearn import metrics

preds_proba_rf = best_rf.predict_proba(X=X_test)
preds_rf = best_rf.predict(X=X_test)

print("Accuracy: ", round(metrics.accuracy_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Precision: ", round(metrics.precision_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Recall: ", round(metrics.recall_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("F1: ", round(metrics.f1_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Confusion matrix: \n", confusion_matrix(y_test, preds_rf))

#%%
#get the roc curve to check if we are better than a random classifier
get_roc_curve(X_train, y_train,X_test, y_test, best_rf)
# %%

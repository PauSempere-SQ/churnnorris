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
#RFE CV to train with less feats
from sklearn import feature_selection as fs 
from sklearn import linear_model as lin 
import matplotlib.pyplot as plt

lr = lin.LogisticRegression(class_weight = "balanced", 
                            solver = "lbfgs", max_iter = 100000)

rfe = fs.RFECV(estimator = lr, step = 1, n_jobs = -1, cv = 5, scoring='f1_weighted', verbose = 2)
rfe.fit(X=X_train, y=y_train.ravel())

print("Optimal number of features : %d" % rfe.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.show()

#%%
X_train_RFE = pd.DataFrame(rfe.transform(X_train), columns = X_train.columns[rfe.support_])
X_test_RFE = pd.DataFrame(rfe.transform(X_test), columns = X_test.columns[rfe.support_])

print(X_train.shape, X_train_RFE.shape)

#%%
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE

#OVERSAMPLING
sampler_smote = SMOTE(n_jobs = -1)
sampler_svm =  SVMSMOTE(n_jobs = -1)
sampler_adasyn = ADASYN(n_jobs = -1)

X_smote, y_smote = sampler_smote.fit_resample(X = X_train, y=y_train.ravel())
X_svm, y_svm = sampler_svm.fit_resample(X = X_train, y=y_train.ravel())
X_adasyn, y_adasyn = sampler_adasyn.fit_resample(X = X_train, y=y_train.ravel())

#%%
X_smote.shape

#%%
#baseline
rf = ensemble.RandomForestClassifier(n_estimators = 100, max_depth=8, criterion="entropy", n_jobs=-1)
rf.fit(X = X_train, y = y_train.ravel())

#FROM NOW ON, USE THE TUNED VERSION ALTHOUGH WE SHOULD RE-TUNE

#weight classes 
rf_balanced = ensemble.RandomForestClassifier(n_jobs=-1,
                                              class_weight = "balanced_subsample", ##VERY IMPORTANT
                                              **best_params_rf)
rf_balanced.fit(X = X_train, y=y_train.ravel())

#weight classes + smote
rf_balanced_smote = ensemble.RandomForestClassifier(n_jobs=-1,
                                                    class_weight = "balanced_subsample", ##VERY IMPORTANT
                                                    **best_params_rf) 
rf_balanced_smote.fit(X = X_smote, y=y_smote.ravel())

#default SMOTE
rf_smote = ensemble.RandomForestClassifier(n_jobs=-1,
                                           **best_params_rf)
rf_smote.fit(X = X_smote, y = y_smote.ravel())

#default SVM SMOTE
rf_svm = ensemble.RandomForestClassifier(n_jobs=-1,
                            **best_params_rf)
rf_svm.fit(X = X_svm, y = y_svm.ravel())

#default ADASYN
rf_adasyn = ensemble.RandomForestClassifier(n_jobs=-1,
                            **best_params_rf)
rf_adasyn.fit(X = X_adasyn, y = y_adasyn.ravel())

#%%
#RFE
rf_reduced = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators = 100, class_weight="balanced_subsample",
                                                max_depth = 8, criterion="entropy") #, **best_params_rf)
rf_reduced.fit(X = X_train_RFE, y=y_train.ravel())

#%%
from sklearn import metrics

##VERY IMPORTANT: ALL MODELS HAVE BEEN TRAINED WITH DIFFERENT DATASETS BUT ALL ARE TESTED AGAINST THE SAME TEST
preds_rf_smote = rf_smote.predict(X=X_test)
preds_rf_svm = rf_svm.predict(X=X_test)
preds_rf_adasyn = rf_adasyn.predict(X=X_test)
preds_rf = rf.predict(X=X_test)
preds_rf_balanced = rf_balanced.predict(X=X_test)
preds_rf_balanced_smote = rf_balanced_smote.predict(X=X_test)

#%%
preds_rf_reduced = rf_reduced.predict(X=X_test_RFE)

#%%
#PRINT RESULTS
print("Accuracy BASELINE: ", round(metrics.accuracy_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Precision BASELINE: ", round(metrics.precision_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Recall BASELINE: ", round(metrics.recall_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("F1 BASELINE: ", round(metrics.f1_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))

print("Accuracy BALANCED:", round(metrics.accuracy_score(y_true = preds_rf_balanced, y_pred = y_test.ravel()), 2))
print("Precision BALANCED:", round(metrics.precision_score(y_true = preds_rf_balanced, y_pred = y_test.ravel()), 2))
print("Recall BALANCED:", round(metrics.recall_score(y_true = preds_rf_balanced, y_pred = y_test.ravel()), 2))
print("F1 BALANCED:", round(metrics.f1_score(y_true = preds_rf_balanced, y_pred = y_test.ravel()), 2))

#%%
print("Accuracy REDUCED: ", round(metrics.accuracy_score(y_true = preds_rf_reduced, y_pred = y_test.ravel()), 2))
print("Precision REDUCED: ", round(metrics.precision_score(y_true = preds_rf_reduced, y_pred = y_test.ravel()), 2))
print("Recall REDUCED: ", round(metrics.recall_score(y_true = preds_rf_reduced, y_pred = y_test.ravel()), 2))
print("F1 REDUCED: ", round(metrics.f1_score(y_true = preds_rf_reduced, y_pred = y_test.ravel()), 2))

#%%
print("Accuracy SMOTE: ", round(metrics.accuracy_score(y_true = preds_rf_smote, y_pred = y_test.ravel()), 2))
print("Precision SMOTE: ", round(metrics.precision_score(y_true = preds_rf_smote, y_pred = y_test.ravel()), 2))
print("Recall SMOTE: ", round(metrics.recall_score(y_true = preds_rf_smote, y_pred = y_test.ravel()), 2))
print("F1 SMOTE: ", round(metrics.f1_score(y_true = preds_rf_smote, y_pred = y_test.ravel()), 2))

print("Accuracy SMOTE SVM: ", round(metrics.accuracy_score(y_true = preds_rf_svm, y_pred = y_test.ravel()), 2))
print("Precision SMOTE SVM: ", round(metrics.precision_score(y_true = preds_rf_svm, y_pred = y_test.ravel()), 2))
print("Recall SMOTE SVM: ", round(metrics.recall_score(y_true = preds_rf_svm, y_pred = y_test.ravel()), 2))
print("F1 SMOTE SVM: ", round(metrics.f1_score(y_true = preds_rf_svm, y_pred = y_test.ravel()), 2))

print("Accuracy ADASYN: ", round(metrics.accuracy_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel()), 2))
print("Precision ADASYN: ", round(metrics.precision_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel()), 2))
print("Recall ADASYN: ", round(metrics.recall_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel()), 2))
print("F1 ADASYN: ", round(metrics.f1_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel()), 2))


#%%
#create objects to plot
#OVERALL SCORE
accuracy_baseline = metrics.accuracy_score(y_true = preds_rf, y_pred = y_test.ravel())
accuracy_smote = metrics.accuracy_score(y_true = preds_rf_smote, y_pred = y_test.ravel())
accuracy_svm = metrics.accuracy_score(y_true = preds_rf_svm, y_pred = y_test.ravel())
accuracy_adasyn = metrics.accuracy_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel())
accuracy_balanced = metrics.accuracy_score(y_true = preds_rf_balanced, y_pred = y_test.ravel())
accuracy_reduced = metrics.accuracy_score(y_true = preds_rf_reduced, y_pred = y_test.ravel())

#PRECISION (CHURNED)
precision_baseline = metrics.precision_score(y_true = preds_rf, y_pred = y_test.ravel())
precision_smote = metrics.precision_score(y_true = preds_rf_smote, y_pred = y_test.ravel())
precision_svm = metrics.precision_score(y_true = preds_rf_svm, y_pred = y_test.ravel())
precision_adasyn = metrics.precision_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel())
precision_balanced = metrics.precision_score(y_true = preds_rf_balanced, y_pred = y_test.ravel())
precision_reduced = metrics.precision_score(y_true = preds_rf_reduced, y_pred = y_test.ravel())

#RECALL (NOT CHURNED)
recall_baseline = metrics.recall_score(y_true = preds_rf, y_pred = y_test.ravel())
recall_smote = metrics.recall_score(y_true = preds_rf_smote, y_pred = y_test.ravel())
recall_svm = metrics.recall_score(y_true = preds_rf_svm, y_pred = y_test.ravel())
recall_adasyn = metrics.recall_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel())
recall_balanced = metrics.recall_score(y_true = preds_rf_balanced, y_pred = y_test.ravel())
recall_reduced = metrics.recall_score(y_true = preds_rf_reduced, y_pred = y_test.ravel())

#F1 (HARMONIC MEAN BETWEEN CHURNED AND NOT CHURNED)
f1_baseline = metrics.f1_score(y_true = preds_rf, y_pred = y_test.ravel())
f1_smote = metrics.f1_score(y_true = preds_rf_smote, y_pred = y_test.ravel())
f1_svm = metrics.f1_score(y_true = preds_rf_svm, y_pred = y_test.ravel())
f1_adasyn = metrics.f1_score(y_true = preds_rf_adasyn, y_pred = y_test.ravel())
f1_balanced = metrics.f1_score(y_true = preds_rf_balanced, y_pred = y_test.ravel())
f1_reduced = metrics.f1_score(y_true = preds_rf_reduced, y_pred = y_test.ravel())

precision_scores = [precision_baseline, precision_balanced, precision_reduced, precision_smote, precision_svm, precision_adasyn]
recall_scores = [recall_baseline, recall_balanced, recall_reduced, recall_smote, recall_svm, recall_adasyn]
f1_scores = [f1_baseline, f1_balanced, f1_reduced, f1_smote, f1_svm, f1_adasyn]
accuracy_scores = [accuracy_baseline, accuracy_balanced, accuracy_reduced, accuracy_smote, accuracy_svm, accuracy_adasyn]
score_names = ["Baseline", "Balanced", "Reduced", "SMOTE", "SMOTE_SVM", "ADASYN"]

d_prec={'Scores':precision_scores, 'Version': score_names}
precision_scores_df = pd.DataFrame(d_prec)

d_f1={'Scores':f1_scores, 'Version': score_names}
f1_scores_df = pd.DataFrame(d_f1)

d_accuracy = {'Scores':accuracy_scores, 'Version': score_names}
accuracy_scores_df = pd.DataFrame(d_accuracy) 

d_recall = {'Scores':recall_scores, 'Version': score_names}
recall_scores_df = pd.DataFrame(d_recall) 
 
#%%
import seaborn as sns

#accuracy plot
sns.set(style="whitegrid")
ax = sns.barplot(x="Scores", y="Version", data = accuracy_scores_df).set_title("Overall scores")

#%%
#precision plot
sns.set(style="whitegrid")
ax = sns.barplot(x="Scores", y="Version", data = precision_scores_df).set_title("Precision scores")

#%%
#recall plot
sns.set(style="whitegrid")
ax = sns.barplot(x="Scores", y="Version", data = recall_scores_df).set_title("Recall scores")

#%%
#f1 plot
sns.set(style="whitegrid")
ax = sns.barplot(x="Scores", y="Version", data = f1_scores_df).set_title("F1 scores")

#%%
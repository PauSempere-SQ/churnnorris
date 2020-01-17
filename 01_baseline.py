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
from IPython.display import HTML
import seaborn as sns

#%%
#load data from our local data folder
base_path = "./data/"
file = "churn_demo.csv"

#drop features that will not be useful, they are either identifiers of the customer or 
#personal information without any relation with their churn likelihood
df_base = pd.read_csv(base_path+file).drop(columns=['RowNumber', 'Surname', 'CustomerId'], axis = 1)

HTML(df_base.head(20).to_html(index = False))

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
#some EDA 

#detect numeric and categorical data
num_cols = list(df_base._get_numeric_data().columns)
num_cols.remove(label)
#use our numerical columns to find out our categorical columns
cat_cols = list(set(df_base.columns) - set(num_cols))

print(num_cols)

#for numeric cols, plot bloxplots hued by the target
for c in num_cols:
    plt.figure(figsize=(8,8)) #create new figure
    ax = sns.boxplot(x=label, y=c,  data=df_base) #plot histogram

#%%
#for categorical data create histograms
print(cat_cols)

for c in cat_cols:
    plt.figure(figsize=(8, 8))
    ax = sns.countplot(x=c, hue=label, data=df_base) 

#%%
#encoders. Just for the sake of the demo, other encoding strategies would be beneficial too
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
#fit a random forest with 100 trees of maximum 8 levels
rf = ensemble.RandomForestClassifier(n_estimators = 100, 
        max_depth=8, criterion="entropy", n_jobs=-1)
rf.fit(X = X_train, y = y_train.ravel())
#%%
#function to plot the ROC curve
def get_roc_curve(X_train, y_train, X_test, y_test, model): 
    probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate AUC
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--', color = "red")
    # plot the precision-recall curve for the model
    plt.plot(fpr, tpr, marker='.', color = "blue")
    # show the plot
    plt.show()

#%%
from sklearn import metrics

#predict using our test set and print different error measures
preds_proba_rf = rf.predict_proba(X=X_test)
preds_rf = rf.predict(X=X_test)

print("Accuracy: ", round(metrics.accuracy_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Precision: ", round(metrics.precision_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Recall: ", round(metrics.recall_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("F1: ", round(metrics.f1_score(y_true = preds_rf, y_pred = y_test.ravel()), 2))
print("Confusion matrix: \n", confusion_matrix(y_test, preds_rf))
#%%
#get the roc curve to check if we are better than a random classifier
get_roc_curve(X_train, y_train,X_test, y_test, rf)

# %%

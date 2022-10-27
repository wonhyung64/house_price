#%%
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# %%
# data_dir= "/Users/wonhyung64/Mirror/house_price"
data_dir= "/Volumes/Lacie/data/house_price"
os.listdir(data_dir)
with open(f"{data_dir}/data_description.txt", "r") as f:
    description = f.readlines()

train_df = pd.read_csv(f"{data_dir}/train.csv")
test_df = pd.read_csv(f"{data_dir}/test.csv")
submit_df = pd.read_csv(f"{data_dir}/sample_submission.csv")

print(train_df.head())
print(test_df.head())
print(submit_df.head())

print(train_df.info())
print(test_df.info())

fig_na, ax = plt.subplots(figsize=(10, 10))
msno.matrix(train_df, ax=ax)

na_ratios = {}
for col in train_df.columns:
    na_ratio = sum(train_df[col].isna())/ len(train_df)
    print(f"{col}: {na_ratio}")
    if na_ratio != 0: 
        na_ratios[col] = na_ratio

na_ratios_test = {}
for col in test_df.columns:
    na_ratio = sum(test_df[col].isna())/ len(test_df)
    print(f"{col}: {na_ratio}")
    if na_ratio != 0: 
        na_ratios_test[col] = na_ratio

fig_dist, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=train_df, x="SalePrice", kde=True, stat="density", ax=ax)

skewness = train_df["SalePrice"].skew()
kurtosis = train_df["SalePrice"].kurt()

useable_cols = list(train_df.columns)
useable_cols.remove("Alley")
useable_cols.remove("PoolQC")
useable_cols.remove("Fence")
useable_cols.remove("MiscFeature")
useable_cols.remove("FireplaceQu")
useable_cols.remove("Id")
train_df = train_df[useable_cols]

useable_cols.remove("SalePrice")
test_df = test_df[useable_cols]

#%% eda
description
train_df.info()

categorical_vars = []
numerical_vars = []
for col in train_df.columns:
    if train_df[col].dtype == "object":
        categorical_vars.append(col)
    else:
        numerical_vars.append(col)

len(categorical_vars)
len(numerical_vars)

nrows, ncols = 6, 3
col_idx = 0
fig_scatter1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 25))
for nrow in range(nrows):
    for ncol in range(ncols):
        sns.scatterplot(data=train_df, y="SalePrice", x=numerical_vars[col_idx], ax=axes[nrow][ncol])
        y_labels = range(0, 800000, 100000)
        plt.sca(axes[nrow][ncol])
        plt.yticks(ticks=y_labels, labels=[f"{y_label / 100000}" for y_label in y_labels])
        col_idx += 1


nrows, ncols = 7, 3
fig_scatter2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 25))
for nrow in range(nrows):
    for ncol in range(ncols):
        sns.scatterplot(data=train_df, y="SalePrice", x=numerical_vars[col_idx], ax=axes[nrow][ncol])
        y_labels = range(0, 800000, 100000)
        plt.sca(axes[nrow][ncol])
        plt.yticks(ticks=y_labels, labels=[f"{y_label / 100000}" for y_label in y_labels])
        col_idx += 1
        if col_idx == 36: break


#%%

nrows, ncols = 6, 3
col_idx = 0
fig_bar1, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 25))
for nrow in range(nrows):
    for ncol in range(ncols):
        sns.barplot(data=train_df, y="SalePrice", x=categorical_vars[col_idx], ax=axes[nrow][ncol])
        col_idx += 1

nrows, ncols = 7, 3
fig_bar2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 25))
for nrow in range(nrows):
    for ncol in range(ncols):
        sns.barplot(data=train_df, y="SalePrice", x=categorical_vars[col_idx], ax=axes[nrow][ncol])
        col_idx += 1

#%%
test_X = test_df[useable_cols]
test_X = test_X.dropna()

useable_cols.append("SalePrice")
X = train_df[useable_cols]
X = X.dropna()


#%%
from tqdm import tqdm
tqdm.pandas()

X["SalePrice_log"] = X["SalePrice"].progress_map(lambda x: np.log(x))
fig_dist_log, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=X, x="SalePrice_log", kde=True, stat="density", ax=ax)

for categorical_var in categorical_vars:
    encoder = LabelEncoder()
    encoder.fit(X[categorical_var].tolist() + test_X[categorical_var].tolist())
    X[categorical_var] = encoder.transform(X[categorical_var])
    test_X[categorical_var] = encoder.transform(test_X[categorical_var])

#%%
res = {}
for max_depth in tqdm(range(2,10)):
    for n_estimators in range(50, 200, 50):
        metrics = []
        for seed in range(10):
            train_x, test_x, train_y, test_y = train_test_split(X.loc[:,:"SaleCondition"], X.loc[:,"SalePrice_log"], random_state=seed)
            gboost = GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators)
            gboost.fit(train_x, train_y)
            pred_y = gboost.predict(test_x)
            metric = mean_squared_error(test_y, pred_y)
            metrics.append(metric)
        res[np.mean(metrics)] = [max_depth, n_estimators]
        
optimal_max_depth, optimal_n_estimators = res[min(res.keys())]

gboost = GradientBoostingRegressor(max_depth=optimal_max_depth, n_estimators=optimal_n_estimators)
gboost.fit(X.loc[:,:"SaleCondition"], X.loc[:,"SalePrice_log"])
pred = gboost.predict(test_X)
np.exp(pred)

test_X["SalePrice"] = np.exp(pred)
test_df
submit_df["SalePrice"] = test_X["SalePrice"]
submit_df = submit_df.fillna(0.)
submit_df.to_csv("/Users/wonhyung64/Downloads/submit.csv", index=False)
os.getcwd()





# %%

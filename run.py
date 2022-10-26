#%%
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# %%
data_dir= "/Users/wonhyung64/data/house_price"
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
        col_idx += 1

nrows, ncols = 7, 3
fig_scatter2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 25))
for nrow in range(nrows):
    for ncol in range(ncols):
        sns.scatterplot(data=train_df, y="SalePrice", x=numerical_vars[col_idx], ax=axes[nrow][ncol])
        col_idx += 1
        if col_idx == 36: break



# %%

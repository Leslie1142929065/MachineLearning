#EDA探索性数据分析
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
# from mlxtend.regressor import StackingCVRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train['SalePrice'].skew())
print(train['SalePrice'].kurt())

corrmat = train.corr()
# corrmat是相关性矩阵
k = 10 #需要多少个相关元素
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index   #返回10个与SalePrice相关性最强的元素的系数
cm = np.corrcoef(train[cols].values.T) #系数矩阵
sns.set(font_scale=0.75)
hm = sns.heatmap(cm, cmap='GnBu', cbar=True,annot =True,square=True,fmt='.2f',annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values )
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']  #选择合适的相关因子绘制Pair图
sns.pairplot(train[cols], height = 2.5)

# data：矩阵数据集，可以使numpy的数组（array），如果是pandas的dataframe，则df的index/column信息会分别对应到heatmap的columns和rows
# vmax,vmin, 图例中最大值和最小值的显示值，没有该参数时默认不显示
# linewidths,热力图矩阵之间的间隔大小
# cmap，热力图颜色
# ax，绘制图的坐标轴，否则使用当前活动的坐标轴。
# annot，annotate的缩写，annot默认为False，当annot为True时，在heatmap中每个方格写入数据。
# annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等：
# sns.heatmap(x, annot=True, ax=ax2, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
# fmt，格式设置，决定annot注释的数字格式，小数点后几位等；
# cbar : 是否画一个颜色条
# cbar_kws : 颜色条的参数，关键字同 fig.colorbar，可以参考：matplotlib自定义colorbar颜色条-以及matplotlib中的内置色条。
# mask，遮罩

# 获取数值型特征
# numeric_features = train.dtypes[train.dtypes != 'object'].index
#
#
# # 计算每个特征的离群样本
# def detect_outliers(x, y, top=5, plot=True):
#     lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)
#     x_ =np.array(x).reshape(-1,1)
#     preds = lof.fit_predict(x_)
#     lof_scr = lof.negative_outlier_factor_
#     out_idx = pd.Series(lof_scr).sort_values()[:top].index
#     if plot:
#         f, ax = plt.subplots(figsize=(9, 6))
#         plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')
#     return out_idx
# all_outliers=[]
# outliers = [30, 88, 462, 523, 632, 1298, 1324]
# for feature in numeric_features:
#     try:
#         outs = detect_outliers(train[feature], train['SalePrice'],top=5, plot=False)
#     except:
#         continue
#     all_outliers.extend(outs)
#     print(Counter(all_outliers).most_common())
#     for i in outliers:
#         if i in all_outliers:
#             print(i)
# for feature in numeric_features:
#     outs = detect_outliers(train[feature], train['SalePrice'],top=5, plot=False)
#     all_outliers.extend(outs)
#
# # 输出离群次数最多的样本
# print(Counter(all_outliers).most_common())
#
# # 剔除离群样本
# train = train.drop(train.index[outliers])
# train.shape
def detect_outliers(x, y, top=5, plot=True):
    lof = LocalOutlierFactor(n_neighbors=40, contamination=0.1)
    x_ = np.array(x).reshape(-1, 1)
    lof.fit_predict(x_)
    lof_scr = lof.negative_outlier_factor_
    out_idx= pd.Series(lof_scr).sort_values()[:top].index
    if plot:
        f, ax = plt.subplots(figsize=(9, 6))
        plt.scatter(x=x, y=y, c=np.exp(lof_scr), cmap='RdBu')
    return out_idx

outs = detect_outliers(train['LowQualFinSF'], train['SalePrice'],top=5) #got 1298,523
print(outs)
outliers = [30, 88, 462, 523, 632, 1298, 1324 ]

all_outliers=[]
numeric_features = train.dtypes[train.dtypes != 'object'].index
for feature in numeric_features:
    try:
        outs = detect_outliers(train[feature], train['SalePrice'], top=5, plot=False)
    except:
        continue
    all_outliers.extend(outs)

print(Counter(all_outliers).most_common())
for i in outliers:
    if i in all_outliers:
        print(i)

train = train.drop(train.index[outliers])
print(train.shape)

y = train.SalePrice.reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)  #axis = 1 代表按列删除
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.drop(['Id'], axis=1, inplace=True)
print(features.shape)

features['MSSubClass'] = features['MSSubClass'].apply(str)
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)
features['OverallQual'] = features['OverallQual'].astype(str)
features['OverallCond'] = features['OverallCond'].astype(str)

numeric_features = features.dtypes[features.dtypes != 'object'].index
print(numeric_features)
print('The number of numeric features is :', len(numeric_features)) #31
category_features = features.dtypes[features.dtypes == 'object'].index
print(category_features)
print('The number of category_features is:', len(category_features)) #48
#总共81个features，前面删除了saleprice & ID

special_features = [
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence'
]
print('The number of special_features is:', len(special_features))




features['Functional'] = features['Functional'].fillna('Typ') #Typ  Typical Functionality
features['Electrical'] = features['Electrical'].fillna("SBrkr") #SBrkr  Standard Circuit Breakers & Romex
features['KitchenQual'] = features['KitchenQual'].fillna("TA")
features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
print(features['MSZoning'])
# plt.show()
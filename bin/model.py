import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

#%matplotlib inline

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

ID = test['Id']
def missing_percent(df):
    missing = df.isna().mean().sort_values(ascending=False).head(20)*100
    missing = pd.DataFrame({'missing %age': missing})
    fig = px.bar(data_frame=missing, x=missing.index, y='missing %age')
    fig.show()
    return missing

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.select_dtypes('object').columns.tolist()
numeric_cols.pop()

#Pre-processing Train data
train['PoolQC'] = np.where(train['PoolQC'].isna(), 'None', train['PoolQC'])
train['MiscFeature'] = np.where(train['MiscFeature'].isna(), 'None', train['MiscFeature'])
train['Alley'] = np.where(train['Alley'].isna(), 'None', train['Alley'])
train['Fence'] = np.where(train['Fence'].isna(), 'None', train['Fence'])
train['FireplaceQu'] = np.where(train['FireplaceQu'].isna(), 'None', train['FireplaceQu'])
train['GarageFinish'] = np.where(train['GarageFinish'].isna(), 'None', train['GarageFinish'])
train['GarageCond'] = np.where(train['GarageCond'].isna(), 'None', train['GarageCond'])
train['GarageType'] = np.where(train['GarageType'].isna(), 'None', train['GarageType'])
train['GarageQual'] = np.where(train['GarageQual'].isna(), 'None', train['GarageQual'])
train['BsmtFinType2'] = np.where(train['BsmtFinType2'].isna(), 'None', train['BsmtFinType2'])
train['BsmtQual'] = np.where(train['BsmtQual'].isna(), 'None', train['BsmtQual'])
train['BsmtCond'] = np.where(train['BsmtCond'].isna(), 'None', train['BsmtCond'])
train['BsmtExposure'] = np.where(train['BsmtExposure'].isna(), 'None', train['BsmtExposure'])
train['BsmtFinType1'] = np.where(train['BsmtFinType1'].isna(), 'None', train['BsmtFinType1'])
train['MasVnrType'] = np.where(train['MasVnrType'].isna(), 'None', train['MasVnrType'])

for i in categorical_cols:
    train[i].fillna(train[i].mode()[0], inplace=True)

for i in numeric_cols:
    ## train
    random = train[i].dropna().sample(train[i].isna().sum())
    random.index = train[train[i].isna()].index
    train[i].fillna(random, inplace=True)

train.isna().sum().sort_values(ascending=False).head()




#Pre-Processing Test Data
test['PoolQC'] = np.where(test['PoolQC'].isna(), 'None', test['PoolQC'])
test['MiscFeature'] = np.where(test['MiscFeature'].isna(), 'None', test['MiscFeature'])
test['Alley'] = np.where(test['Alley'].isna(), 'None', test['Alley'])
test['Fence'] = np.where(test['Fence'].isna(), 'None', test['Fence'])
test['FireplaceQu'] = np.where(test['FireplaceQu'].isna(), 'None', test['FireplaceQu'])
test['GarageFinish'] = np.where(test['GarageFinish'].isna(), 'None', test['GarageFinish'])
test['GarageCond'] = np.where(test['GarageCond'].isna(), 'None', test['GarageCond'])
test['GarageType'] = np.where(test['GarageType'].isna(), 'None', test['GarageType'])
test['GarageQual'] = np.where(test['GarageQual'].isna(), 'None', test['GarageQual'])
test['BsmtFinType2'] = np.where(test['BsmtFinType2'].isna(), 'None', test['BsmtFinType2'])
test['BsmtQual'] = np.where(test['BsmtQual'].isna(), 'None', test['BsmtQual'])
test['BsmtCond'] = np.where(test['BsmtCond'].isna(), 'None', test['BsmtCond'])
test['BsmtExposure'] = np.where(test['BsmtExposure'].isna(), 'None', test['BsmtExposure'])
test['BsmtFinType1'] = np.where(test['BsmtFinType1'].isna(), 'None', test['BsmtFinType1'])
test['MasVnrType'] = np.where(test['MasVnrType'].isna(), 'None', test['MasVnrType'])

for i in categorical_cols:
    test[i].fillna(test[i].mode()[0], inplace=True)

for i in numeric_cols:
    ## test
    random = test[i].dropna().sample(test[i].isna().sum())
    random.index = test[test[i].isna()].index
    test[i].fillna(random, inplace=True)

test.isna().sum().sort_values(ascending=False).head()


#Featuring DataSets

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(train[categorical_cols])
encoded_cols = encoder.get_feature_names(categorical_cols)

test[encoded_cols] = encoder.transform(test[categorical_cols])
test.drop(categorical_cols, axis=1, inplace=True)

train[encoded_cols] = encoder.transform(train[categorical_cols])
train.drop(categorical_cols, axis=1, inplace=True)

df = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)

skewness = df[numeric_cols].skew().sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]

for i in skewness.index:
    df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

df[skewness.index].skew().sort_values(ascending=False)


#Modeling

y = train['SalePrice']
X = df.iloc[:len(train), :]
test = df.iloc[len(train):, :]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

ridge = Ridge()
param_grid = {
    'alpha': [12, 12.1, 12.2, 12.3, 11.9, 11.8, 11.7, 11.75],
}
ridge_grid = GridSearchCV(ridge, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10).fit(X_train, y_train)
def evaluate(model, X_train, y_train, X_test, y_test):
    print('TRAIN')
    pred = model.predict(X_train)
    print(f'MEAN ABSOLUTE ERROR: {mean_absolute_error(y_train, pred)}')
    print(f'MEAN SQUARED ERROR: {mean_squared_error(y_train, pred)}')
    print(f'ROOT MEAN SQUARED ERROR: {np.sqrt(mean_squared_error(y_train, pred))}')
    print(f'R2 SCORE: {r2_score(y_train, pred)}')
    print('------------------------------------------------------')
    print('TEST')
    pred = model.predict(X_test)
    print(f'MEAN ABSOLUTE ERROR: {mean_absolute_error(y_test, pred)}')
    print(f'MEAN SQUARED ERROR: {mean_squared_error(y_test, pred)}')
    print(f'ROOT MEAN SQUARED ERROR: {np.sqrt(mean_squared_error(y_test, pred))}')
    print(f'R2 SCORE: {r2_score(y_test, pred)}')
evaluate(ridge_grid, X_train, y_train, X_test, y_test)



def preproc(test):
    print("Inside Preprocessing:")

    print(test)

    test.drop('Id', axis=1, inplace=True)

    numeric_cols = test.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = test.select_dtypes('object').columns.tolist()
    numeric_cols.pop()

    test['PoolQC'] = np.where(test['PoolQC'].isna(), 'None', test['PoolQC'])
    test['MiscFeature'] = np.where(test['MiscFeature'].isna(), 'None', test['MiscFeature'])
    test['Alley'] = np.where(test['Alley'].isna(), 'None', test['Alley'])
    test['Fence'] = np.where(test['Fence'].isna(), 'None', test['Fence'])
    test['FireplaceQu'] = np.where(test['FireplaceQu'].isna(), 'None', test['FireplaceQu'])
    test['GarageFinish'] = np.where(test['GarageFinish'].isna(), 'None', test['GarageFinish'])
    test['GarageCond'] = np.where(test['GarageCond'].isna(), 'None', test['GarageCond'])
    test['GarageType'] = np.where(test['GarageType'].isna(), 'None', test['GarageType'])
    test['GarageQual'] = np.where(test['GarageQual'].isna(), 'None', test['GarageQual'])
    test['BsmtFinType2'] = np.where(test['BsmtFinType2'].isna(), 'None', test['BsmtFinType2'])
    test['BsmtQual'] = np.where(test['BsmtQual'].isna(), 'None', test['BsmtQual'])
    test['BsmtCond'] = np.where(test['BsmtCond'].isna(), 'None', test['BsmtCond'])
    test['BsmtExposure'] = np.where(test['BsmtExposure'].isna(), 'None', test['BsmtExposure'])
    test['BsmtFinType1'] = np.where(test['BsmtFinType1'].isna(), 'None', test['BsmtFinType1'])
    test['MasVnrType'] = np.where(test['MasVnrType'].isna(), 'None', test['MasVnrType'])

    for i in categorical_cols:
        test[i].fillna(test[i].mode()[0], inplace=True)

    for i in numeric_cols:
        ## test
        random = test[i].dropna().sample(test[i].isna().sum())
        random.index = test[test[i].isna()].index
        test[i].fillna(random, inplace=True)
    test.isna().sum().sort_values(ascending=False).head()

    test[encoded_cols] = encoder.transform(test[categorical_cols])
    test.drop(categorical_cols, axis=1, inplace=True)

    df = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    skewness = skewness[abs(skewness) > 0.5]
    for i in skewness.index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))
    df[skewness.index].skew().sort_values(ascending=False)

    predict(test)





######
def predict(test):

    print("Predicting Model for given data :", test)
    pred = ridge_grid.predict(test)
    submission = pd.DataFrame({
        'Id': ID,
        'SalePrice': pred
    })
    print("Prediction below:")
    print(submission)

#print(type(test))
predict(test)

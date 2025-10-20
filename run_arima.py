import warnings
warnings.filterwarnings('ignore')


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------- User parameters --------------------
DATA_PATH = '/home/koushik/CATS_CaseStudy/dataset/nift/NIFTY_COMMODITIES_minute.csv' # <- update path
DATETIME_COL = 'date'
TARGET = 'close'
EXOG_VARS = ['open','high','low']
RESAMPLE_RULE = None # e.g. '5T' for 5-min, '15T', '1H' or None to keep original
TEST_SIZE_MINUTES = 3000 # number of last rows to keep as test (or set train/test fraction)
MAX_P = 3
MAX_D = 2
MAX_Q = 3
RANDOM_STATE = 42
RESULT_DIR = "/"

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

def resample_df(df, rule):
    if rule is None:
        return df
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        # 'volume': 'sum'
    }
    return df.resample(rule).agg(agg).dropna()

df.reset_index(drop=True, inplace=True)
df = df.set_index("date", drop=True)
df = resample_df(df, "H")
df.reset_index(drop=False, inplace=True)
print(df.head())

lag_features = EXOG_VARS
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index().astype(np.float32)
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index().astype(np.float32)
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index().astype(np.float32)

df_std_3d = df_rolled_3d.std().shift(1).reset_index().astype(np.float32)
df_std_7d = df_rolled_7d.std().shift(1).reset_index().astype(np.float32)
df_std_30d = df_rolled_30d.std().shift(1).reset_index().astype(np.float32)

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("date", drop=False, inplace=True)
# df = df.asfreq('15T').ffill()

df_train = df[(df['date'] > '2022-01-01') & (df['date'] < '2024-01-01')]
df_valid = df[df.date >= "2024"]

exogenous_features = ['open_mean_lag3', 'open_mean_lag7', 'open_mean_lag30', 'open_std_lag3', 'open_std_lag7',
       'open_std_lag30', 'high_mean_lag3', 'high_mean_lag7', 'high_mean_lag30',
       'high_std_lag3', 'high_std_lag7', 'high_std_lag30', 'low_mean_lag3',
       'low_mean_lag7', 'low_mean_lag30', 'low_std_lag3', 'low_std_lag7',
       'low_std_lag30']

train_exog = df_train[exogenous_features]
train_exog = train_exog.apply(pd.to_numeric, errors='coerce')
# train_exog.reset_index(drop=True, inplace=True)



print(train_exog.head())

print('\n6) Grid-searching ARIMA(p,d,q) by AIC (this can take time)')
best_aic = np.inf
best_order = None
best_model = None
for p in range(0, MAX_P+1):
    for d in range(0, MAX_D+1):
        for q in range(0, MAX_Q+1):
            try:
                # print(f'order=({p},{d},{q})')
                model = SARIMAX(df_train.close, exog=train_exog, order=(p,d,q), enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False, maxiter=100)
                aic = res.aic
                print(f'order=({p},{d},{q}) AIC={aic:.2f}')
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p,d,q)
                    best_model = res
            except Exception as e:
                print(f'order=({p},{d},{q}) failed: {e}')
                continue

print(f'Best order by AIC: {best_order} with AIC={best_aic:.2f}')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import holidays
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# Load datasets
train = pd.read_csv('/content/train.csv', parse_dates=['doj'])
test = pd.read_csv('/content/test_8gqdJqH.csv', parse_dates=['doj'])
transactions = pd.read_csv('/content/transactions.csv', parse_dates=['doj', 'doi'])

# Filter transactions 15 days before journey
trans_15 = transactions[transactions['dbd'] == 15].copy()

# Load Indian holidays
indian_holidays = holidays.India(years=range(2018, 2026))

# Create features
def create_features(df):
    df['doj_weekday'] = df['doj'].dt.weekday
    df['doj_month'] = df['doj'].dt.month
    df['weekend'] = df['doj_weekday'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['doj'].dt.day <= 3).astype(int)
    df['src_dest'] = df['srcid'].astype(str) + '_' + df['destid'].astype(str)

    df['is_holiday'] = df['doj'].isin(indian_holidays).astype(int)
    df['prev_day_is_holiday'] = (df['doj'] - timedelta(days=1)).isin(indian_holidays).astype(int)
    df['next_day_is_holiday'] = (df['doj'] + timedelta(days=1)).isin(indian_holidays).astype(int)
    df['long_weekend'] = (
        ((df['doj_weekday'] == 4) & df['next_day_is_holiday']) |
        ((df['doj_weekday'] == 0) & df['prev_day_is_holiday'])
    ).astype(int)

    return df

# Apply feature creation
train = create_features(train)
test = create_features(test)
trans_15 = create_features(trans_15)

# Aggregate transaction data
agg_trans = trans_15.groupby(['doj', 'srcid', 'destid']).agg({
    'cumsum_seatcount': ['sum', 'mean'],
    'cumsum_searchcount': ['sum', 'mean']
}).reset_index()
agg_trans.columns = ['doj', 'srcid', 'destid',
                     'cumsum_seatcount', 'seat_mean',
                     'cumsum_searchcount', 'search_mean']
agg_trans = create_features(agg_trans)

# Merge aggregated data
train = pd.merge(train, agg_trans, on=['doj', 'srcid', 'destid', 'doj_weekday', 'doj_month'], how='left')
test = pd.merge(test, agg_trans, on=['doj', 'srcid', 'destid', 'doj_weekday', 'doj_month'], how='left')

# Fill missing values
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

# ✅ Fix: recompute src_dest after merge
train['src_dest'] = train['srcid'].astype(str) + '_' + train['destid'].astype(str)
test['src_dest'] = test['srcid'].astype(str) + '_' + test['destid'].astype(str)

# Add derived features
def add_derived_features(df):
    df['seat_to_search_ratio'] = np.where(df['cumsum_searchcount'] > 0,
                                          df['cumsum_seatcount'] / df['cumsum_searchcount'], 0)
    df['route_freq'] = df.groupby('src_dest')['src_dest'].transform('count')
    df['route_diff'] = abs(df['srcid'] - df['destid'])
    df['demand_category'] = pd.qcut(df['cumsum_seatcount'].rank(method='first'), q=4, labels=False, duplicates='drop')
    return df
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

train['src_dest'] = train['srcid'].astype(str) + '_' + train['destid'].astype(str)
test['src_dest'] = test['srcid'].astype(str) + '_' + test['destid'].astype(str)

train = create_features(train)
test = create_features(test)

train = add_derived_features(train)
test = add_derived_features(test)


features = [
    'srcid', 'destid', 'doj_weekday', 'doj_month', 'weekend', 'is_month_start',
    'is_holiday', 'prev_day_is_holiday', 'next_day_is_holiday', 'long_weekend',
    'cumsum_seatcount', 'seat_mean',
    'cumsum_searchcount', 'search_mean',
    'seat_to_search_ratio', 'route_freq', 'route_diff', 'demand_category'
]
target = 'final_seatcount'

X = train[features]
y = train[target]
X_test = test[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.03,
    n_estimators=1500,
    num_leaves=64,
    feature_fraction=0.85,
    bagging_fraction=0.7,
    bagging_freq=5,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(100)
    ]
)

lgb_val_pred = lgb_model.predict(X_val)
lgb_test_pred = lgb_model.predict(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

xgb_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'seed': 42
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=1500,
    evals=[(dtrain, 'train'), (dval, 'valid')],
    early_stopping_rounds=100,
    verbose_eval=100
)

xgb_val_pred = xgb_model.predict(dval)
xgb_test_pred = xgb_model.predict(dtest)

val_pred = 0.6 * lgb_val_pred + 0.4 * xgb_val_pred
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"✅ Validation RMSE (Ensemble): {rmse:.4f}")

test_preds = 0.6 * lgb_test_pred + 0.4 * xgb_test_pred

submission = pd.DataFrame({
    'route_key': test['route_key'],
    'final_seatcount': test_preds
})
submission.to_csv('submission.csv', index=False)
print("🎯 Submission file saved as 'submission.csv'")


from fbprophet import Prophet
import pandas as pd
import numpy as np
%matplotlib inline
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
FULL_df = pd.read_csv('train.csv', ',', index_col=0, parse_dates=[1], date_parser=dateparse)
all_df = []
max(np.array(FULL_df['tsID']))

for i in range(1, max(np.array(FULL_df['tsID'])) + 1):
    df = FULL_df[FULL_df['tsID']==i]
    train_ratio = 0.2
    old_colmns = df.columns
    df['ds'] = df['Date']
    df['y']=  df['ACTUAL']
    # delete test from data
    not_nulls_ids = df['ACTUAL'].notnull()
    nulls_ids = df['ACTUAL'].isnull()
    kaggle_df = df.drop(old_colmns, axis=1)[nulls_ids]
    df = df.drop(old_colmns, axis=1)[not_nulls_ids]
    test_size = int(train_ratio*(len(df)))
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    model = Prophet()
    model.fit(df)
    future_data = test_df[['ds']]#model.make_future_dataframe(periods=12, freq = 'm')
    kaggle_data = kaggle_df[['ds']]
    forecast_data = model.predict(kaggle_data)
    # model.plot(forecast_data)
    kaggle_df['ID'] = kaggle_df.index
    cur_df = pd.merge(kaggle_df, forecast_data[['ds','yhat']], how='inner', left_on='ds', right_on='ds',)
    all_df.append(cur_df[['ID', 'yhat']].set_index('ID'))

# %%
result_kaggle = pd.concat(all_df)
result_kaggle['PREDICTED'] = result_kaggle['yhat']
result_kaggle = result_kaggle[['PREDICTED']]
result_kaggle.to_csv('result.csv')
#
# # %%
# # forecast_data
# cmp_df = forecast_data.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(df.set_index('ds'))
# cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
# cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']
# print( 'MAPE', np.mean(abs(cmp_df[-test_size:]['p'])))
# print( 'MAE', np.mean(abs(cmp_df[-test_size:]['e'])))
# # %%
# # forecast_data.columns
# forecast_data[['ds', 'yhat']]

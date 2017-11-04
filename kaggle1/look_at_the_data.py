# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import plotly.plotly as py
import plotly
import plotly.graph_objs as go

from datetime import datetime

%matplotlib inline

# %%
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')
df = pd.read_csv('train.csv', ',', index_col=0, parse_dates=[1], date_parser=dateparse)
df = df[df['tsID']==1]
# df.set_index('Date').plot(figsize=(20, 5))
df.set_index('Date')['ACTUAL'].plot(figsize=(20, 5))
# df
# %%
df['Date'].head(5)

# %%
nulls_ids = np.array(df[df['ACTUAL'].isnull()].index)
not_nulls_ids = np.array(1 - df[ df['ACTUAL'].isnull()].index)
plt.figure(figsize=(15, 1))
plt.scatter(nulls_ids, np.zeros(len(nulls_ids)))
plt.scatter(not_nulls_ids, np.zeros(len(not_nulls_ids)))
plt.show()
# %%



not_null_df = df[df['ACTUAL'].notnull()]
two_values = not_null_df[['Date', 'ACTUAL']]
ser = two_values.set_index('Date')
# ser.plot(figsize=(20, 8))
data = [go.Scatter(x=two_values['Date'], y=two_values['ACTUAL'])]

plotly.offline.plot(data)

# %%

# ddf = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
# del ddf

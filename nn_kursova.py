# -*- coding: utf-8 -*-
"""NN_kursova.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jQI57UwTzYpnbR3JLH3oSDrNb9HhoGtN

# Import libraries:
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

"""# Dataset:"""

df = pd.read_csv("/dataset.csv")
new_df = df.loc[(df['rpt_key'] == 'btc_usd')]
new_df

new_df = new_df.reset_index(drop=True)
new_df['datetime'] = pd.to_datetime(new_df['datetime_id'])

new_df = new_df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]
new_df

new_df = new_df[['last']]
dataset = new_df.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = train_test_split(dataset, test_size=test_size, shuffle=False, random_state=42)
print(len(train), len(test))

def create_dataset(dataset, prev_pred=1):
    x = []
    y = []
    
    for i in range(len(dataset) - prev_pred - 1):
        a = dataset[i:(i + prev_pred), 0]
        x.append(a)
        y.append(dataset[i + prev_pred, 0])
    
    x = np.array(x)
    y = np.array(y)
    
    return x, y

"""#  Linear Regression:"""

prev_pred = 10
x_train, y_train = create_dataset(train, prev_pred=prev_pred)
x_test, y_test = create_dataset(test, prev_pred=prev_pred)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2]))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

regressor = LinearRegression()
regressor.fit(x_test, y_test)

linear_pred = regressor.predict(x_test)

linear_pred = scaler.inverse_transform(linear_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test)

mae_linear = mean_absolute_error(y_test, linear_pred)
r2_linear = r2_score(y_test, linear_pred)
variance_linear = explained_variance_score(y_test, linear_pred)

print("MAE: ", mae_linear)
print("R2-score: ", r2_linear)
print("Variance score: ", variance_linear)

result_LR = pd.DataFrame({'Actual': y_test.flatten(), 'LR_pred': linear_pred.flatten()})
result_LR['Difference'] = result_LR['LR_pred'] - result_LR['Actual']
result_LR

fig_LR = go.Figure()
fig_LR.add_trace(go.Scatter(x=result_LR.index, y=result_LR['Actual'], mode='lines', name='Actual'))
fig_LR.add_trace(go.Scatter(x=result_LR.index, y=result_LR['LR_pred'], mode='lines', name='LR_pred'))

fig_LR.update_layout(
    title='Actual vs Predicted for Linear Regression',
    xaxis_title='Index',
    yaxis_title='Value',
    showlegend=True,
    legend=dict(x=0, y=1),
)

fig_LR.show()

train_sizes, train_scores, test_scores = learning_curve(regressor, x_test, y_test, cv=5, scoring='neg_mean_absolute_error')

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

fig, ax = plt.subplots()
ax.plot(train_sizes, train_scores_mean, label='Training error')
ax.plot(train_sizes, test_scores_mean, label='Test error')
ax.set_xlabel('Training set size')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Learning Curves')
ax.legend()
plt.show()

"""## LSTM"""

prev_pred = 10
x_train, y_train = create_dataset(train, prev_pred=prev_pred)
x_test, y_test = create_dataset(test, prev_pred=prev_pred)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

model = Sequential()
model.add(LSTM(4, input_shape=(1, prev_pred)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=2)

pred_LSTM = model.predict(x_test)

pred_LSTM = scaler.inverse_transform(pred_LSTM)
y_test = scaler.inverse_transform([y_test])

mae_test = mean_absolute_error(y_test[0], pred_LSTM[:, 0])
r2_test = r2_score(y_test[0], pred_LSTM[:, 0])
variance_test = explained_variance_score(y_test[0], pred_LSTM[:, 0])

print("MAE: ", mae_test)
print("R2-score: ", r2_test)
print("Variance score: ", variance_test)

result_LSTM = pd.DataFrame({'Actual': y_test.flatten(), 'pred_LSTM': pred_LSTM.flatten()})
result_LSTM['Difference'] = result_LSTM['pred_LSTM'] - result_LSTM['Actual']
result_LSTM

fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=result_LSTM.index, y=result_LSTM['Actual'], mode='lines', name='Actual'))
fig_lstm.add_trace(go.Scatter(x=result_LSTM.index, y=result_LSTM['pred_LSTM'], mode='lines', name='pred_LSTM'))

fig_lstm.update_layout(
    title='Actual vs Predicted for LSTM',
    xaxis_title='Index',
    yaxis_title='Value',
    showlegend=True,
    legend=dict(x=0, y=1),
)

fig_lstm.show()

fig_difference = go.Figure()
fig_difference.add_trace(go.Scatter(x=result_LSTM.index, y=result_LSTM['Difference'], mode='lines', name='Difference'))

fig_difference.update_layout(
    title='Difference',
    xaxis_title='Index',
    yaxis_title='Difference',
    showlegend=True,
    legend=dict(x=0, y=1),
)

fig_difference.show()

"""# Random Forest:"""

prev_pred = 10
x_train, y_train = create_dataset(train, prev_pred=prev_pred)
x_test, y_test = create_dataset(test, prev_pred=prev_pred)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

rf_model = RandomForestRegressor()

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

rf_model.fit(x_train, y_train)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

rf_pred = rf_model.predict(x_test)

rf_pred = scaler.inverse_transform(rf_pred.reshape(-1, 1)).flatten()
y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae_rf = mean_absolute_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)
variance_rf = explained_variance_score(y_test, rf_pred)

print("Random Forest Results:")
print("MAE: ", mae_rf)
print("R2-score: ", r2_rf)
print("Variance score: ", variance_rf)

result_RF = pd.DataFrame({'Actual': y_test.flatten(), 'RF_pred': rf_pred.flatten()})
result_RF['Difference'] = result_RF['RF_pred'] - result_RF['Actual']
result_RF

fig_RF = go.Figure()
fig_RF.add_trace(go.Scatter(x=result_RF.index, y=result_RF['Actual'], mode='lines', name='Actual'))
fig_RF.add_trace(go.Scatter(x=result_RF.index, y=result_RF['RF_pred'], mode='lines', name='RF_pred'))

fig_RF.update_layout(
    title='Actual vs Predicted for Random Forest',
    xaxis_title='Index',
    yaxis_title='Value',
    showlegend=True,
    legend=dict(x=0, y=1),
)

fig_RF.show()

train_sizes, train_scores, test_scores = learning_curve(rf_model, x_test, y_test, cv=5, scoring='neg_mean_absolute_error')

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

fig, ax = plt.subplots()
ax.plot(train_sizes, train_scores_mean, label='Training error')
ax.plot(train_sizes, test_scores_mean, label='Test error')
ax.set_xlabel('Training set size')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Learning Curves')
ax.legend()
plt.show()

"""# Comparison of model results:"""

result = pd.DataFrame({'Actual': y_test.flatten(), 'LR_pred': linear_pred.flatten(), 
                       'LSTM_pred': pred_LSTM.flatten(), 'RF_pred': rf_pred.flatten()})
result

fig = go.Figure()
fig.add_trace(go.Scatter(x=result.index, y=result['Actual'], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=result.index, y=result['LR_pred'], mode='lines', name='LR_pred'))
fig.add_trace(go.Scatter(x=result.index, y=result['LSTM_pred'], mode='lines', name='LSTM_pred'))
fig.add_trace(go.Scatter(x=result.index, y=result['RF_pred'], mode='lines', name='RF_pred'))

fig.update_layout(
    title='Actual vs Predicted for Linear Regression',
    xaxis_title='Index',
    yaxis_title='Value',
    showlegend=True,
    legend=dict(x=0, y=1),
)

fig.show()
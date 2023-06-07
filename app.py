from flask import Flask, request, render_template
import math
import numpy as np
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.model_selection import learning_curve
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

#python app.py

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            model = request.form['model']
            # Обробка датасету та прогнозу
            new_df = df.loc[(df['rpt_key'] == 'btc_usd')]
            new_df = new_df.reset_index(drop=True)
            new_df['datetime'] = pd.to_datetime(new_df['datetime_id'])
            new_df = new_df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]
            new_df = new_df[['last']]
            dataset = new_df.values
            dataset = dataset.astype('float32')
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train_size = int(len(dataset) * 0.7)
            test_size = len(dataset) - train_size
            train, test = train_test_split(dataset, test_size=test_size, shuffle=False, random_state=42)

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

            prev_pred = 10
            x_train, y_train = create_dataset(train, prev_pred=prev_pred)
            x_test, y_test = create_dataset(test, prev_pred=prev_pred)

            x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2]))
            y_test = np.reshape(y_test, (y_test.shape[0], 1))

            if model == 'linear':
                regressor = joblib.load('linear_regression_model.pkl')
                linear_pred = regressor.predict(x_test)

                linear_pred = scaler.inverse_transform(linear_pred.reshape(-1, 1))
                y_test = scaler.inverse_transform(y_test)

                mae_linear = mean_absolute_error(y_test, linear_pred)
                r2_linear = r2_score(y_test, linear_pred)
                variance_linear = explained_variance_score(y_test, linear_pred)

                result = pd.DataFrame({'Actual': y_test.flatten(), 'Prediction': linear_pred.flatten()})
                result['Difference'] = result['Actual'] - result['Prediction']

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result.index, y=result['Actual'], name='Actual'))
                fig.add_trace(go.Scatter(x=result.index, y=result['Prediction'], name='Linear Regression Predicted'))
                fig.update_layout(title='Actual vs Linear Regression Predicted',
                                  xaxis_title='Index',
                                  yaxis_title='Price')
                graph_data = fig.to_html(full_html=False)

            elif model == 'lstm':
                model = load_model('trained_lstm_model.h5')
                x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
                lstm_pred = model.predict(x_test)

                lstm_pred = scaler.inverse_transform(lstm_pred)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                mae_lstm = mean_absolute_error(y_test, lstm_pred)
                r2_lstm = r2_score(y_test, lstm_pred)
                variance_lstm = explained_variance_score(y_test, lstm_pred)

                result = pd.DataFrame({'Actual': y_test.flatten(), 'Prediction': lstm_pred.flatten()})
                result['Difference'] = result['Actual'] - result['Prediction']

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result.index, y=result['Actual'], name='Actual'))
                fig.add_trace(go.Scatter(x=result.index, y=result['Prediction'], name='LSTM Predicted'))
                fig.update_layout(title='Actual vs LSTM Predicted',
                                  xaxis_title='Index',
                                  yaxis_title='Price')
                graph_data = fig.to_html(full_html=False)

            elif model == 'random_forest':
                regressor = joblib.load('trained_rf_model.joblib')
                rf_pred = regressor.predict(x_test)

                rf_pred = scaler.inverse_transform(rf_pred.reshape(-1, 1))
                y_test = scaler.inverse_transform(y_test)

                mae_rf = mean_absolute_error(y_test, rf_pred)
                r2_rf = r2_score(y_test, rf_pred)
                variance_rf = explained_variance_score(y_test, rf_pred)

                result = pd.DataFrame({'Actual': y_test.flatten(), 'Prediction': rf_pred.flatten()})
                result['Difference'] = result['Actual'] - result['Prediction']

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result.index, y=result['Actual'], name='Actual'))
                fig.add_trace(go.Scatter(x=result.index, y=result['Prediction'], name='Random Forest Predicted'))
                fig.update_layout(title='Actual vs Random Forest Predicted',
                                  xaxis_title='Index',
                                  yaxis_title='Price')
                graph_data = fig.to_html(full_html=False)

            return render_template('result.html', graph=graph_data)

    return render_template('upload.html')


if __name__ == '__main__':
    app.run()

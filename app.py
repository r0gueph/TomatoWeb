# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# import numpy as np

# def load_data():
#     # Step 1: Load the data
#     data = pd.read_csv('AreaHarvested.csv')
#     # Convert 'Year' column to string type
#     data['Year'] = data['Year'].astype(str)
#     # Create a new column combining 'Year' and 'TimePeriod'
#     data['YearQuarter'] = data['Year'] + '-' + data['TimePeriod']
#     return data

# def plot_time_series(data):
#     # Plot the time series data
#     plt.figure(figsize=(12, 6))
#     plt.plot(data['YearQuarter'], data['AreaHarvested'])
#     plt.xlabel('Year')
#     plt.ylabel('Area Harvested')
#     plt.title('Quarterly Area Harvested 2012-2022')
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
#     plt.show()

# # Take the logarithm of the 'AreaHarvested' column
# def take_log(data):
#     data['AreaHarvested_log'] = np.log(data['AreaHarvested'])
#     return data

# # Plot the logarithm of the time series data
# def plot_log_time_series(data):
#     plt.figure(figsize=(12, 6))
#     plt.plot(data['YearQuarter'], data['AreaHarvested_log'])
#     plt.xlabel('Year')
#     plt.ylabel('Area Harvested')
#     plt.title('Differenced Quarterly Area Harvested 2012-2022')
#     plt.xticks(rotation=90)
#     plt.show()

# def check_stationarity(data):
#     train_data = data['AreaHarvested_log'].iloc[:int(len(data) * 0.7)]
#     # Perform the ADF test
#     print('First ADF Test:')
#     result = adfuller(train_data)    
#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print(f'{key}:{value}')

#     train_data_diff = train_data.diff().dropna()
#     # Perform the 2nd ADF test
#     print('\nSecond ADF Test(after differencing):')
#     result = adfuller(train_data_diff)
#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print(f'{key}:{value}')
#     # Print ADF test results

#     return train_data_diff

# # Fit the ARIMA model
# def fit_arima(train_data):
#     model = ARIMA(train_data, order=(4, 1, 0))
#     model_fit = model.fit()
#     print(model_fit.summary())
#     return model_fit

# # Make time series predictions
# def make_predictions(data, model_fit):
#     test_data = data['AreaHarvested_log'].iloc[int(len(data) * 0.7):]
#     forecast = model_fit.forecast(steps=len(test_data))

#     combined_data = pd.concat([data[['YearQuarter', 'AreaHarvested_log']], pd.Series(forecast)], axis=1)
#     combined_data.columns = ['Year', 'Actual', 'Forecast']

#     plt.figure(figsize=(12, 6))
#     plt.plot(combined_data['Year'], combined_data['Actual'], label='Actual')
#     plt.plot(combined_data['Year'], combined_data['Forecast'], label='Test Forecast')
#     plt.xlabel('Year')
#     plt.ylabel('Area Harvested')
#     plt.title('Actual vs Test Forecast Data')
#     plt.xticks(rotation=90)
#     plt.legend()
#     plt.show()

#     return test_data, forecast


# # Evaluate model predictions
# def evaluate_predictions(test_data, forecast):
#     from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
#     # Calculate MAE, MAPE, RMSE
#     mae = mean_absolute_error(test_data, forecast)
#     mape = mean_absolute_percentage_error(test_data, forecast)
#     rmse = np.sqrt(mean_squared_error(test_data, forecast))
    
#     # Print the results
#     print('\nMean Absolute Error (MAE):', mae)
#     print('Mean Absolute Percentage Error (MAPE):', mape)
#     print('Root Mean Squared Error (RMSE):', rmse)

# # Make actual predictions
# def make_actual_predictions(data, model_fit, num_years):
#     num_years = int(input('\nEnter the number of years ahead to predict: '))

#     train_data = data['AreaHarvested'].iloc[:int(len(data) * 0.7)]
#     model = ARIMA(train_data, order=(4, 1, 0))
#     model_fit = model.fit()

#     last_year = int(data['Year'].iloc[-1])
#     last_year = last_year + 1
#     future_years = pd.date_range(start=f'{last_year}-01-01', periods=num_years * 4, freq='Q')
#     future_forecast = pd.Series(model_fit.forecast(steps=num_years * 4))
#     prediction_df = pd.DataFrame({'Year': future_years.year, 'TimePeriod': future_years.quarter, 'Prediction': future_forecast})

#     print('\nForecasted Future Data')
#     print(prediction_df)

#     return prediction_df


from flask import Flask, render_template, request

app = Flask(__name__,static_folder='static')

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/load_data')
# def load_data_endpoint():
#     data = load_data()
#     return render_template('index.html', data=data)

# @app.route('/plot_time_series')
# def plot_time_series_endpoint():
#     data = load_data()
#     plot_time_series(data)
#     return render_template('index.html')

# @app.route('/submit', methods=['POST'])
# def submit(prediction_df):
#     #Process the form submission and generate the results
#      max_area = request.form.get['max_area']
#      forecast_years = request.form.get('forecast_years')

#     # Pass the results to the template for rendering
#      forecast_results = {
#      'max_area': max_area,
#      'forecast_years': forecast_years,
#      'prediction_df': prediction_df.to_dict('records')
#     }
#      return render_template('index.html', forecast_results=forecast_results)

if __name__ == '__main__':
    app.run(debug=True)

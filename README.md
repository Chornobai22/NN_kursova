# Bitcoin price prediction
## Description

This code is designed to predict the price of Bitcoin using various machine learning models. It performs data preprocessing, model training, evaluation, and visualization.

## Dataset

The dataset is stored in a file named "dataset.csv". It reads the dataset and performs various preprocessing steps.

## Main technologies

| Technology   | Version  |
|--------------|----------|
| NumPy        | 1.21.0   |
| pandas       | 1.3.0    |
| matplotlib   | 3.4.2    |
| plotly       | 5.3.1    |
| scikit-learn | 0.24.2   |
| Keras        | 2.6.0    |
| Python       | 3.6.0    |


## Model Training and Evaluation

The code trains and evaluates three different models: Linear Regression, LSTM (Long Short-Term Memory), and Random Forest. The steps involved are as follows:

**Linear Regression:**

• Preprocess the data.

• Split the data into training and testing sets.

• Fit a linear regression model on the training data.

• Evaluate the model's performance using mean absolute error (MAE), R2-score, and variance score.

• Visualize the actual vs. predicted values using a line plot.

**LSTM (Long Short-Term Memory):**

• Preprocess the data.

• Split the data into training and testing sets.

• Build an LSTM model with one LSTM layer.

• Compile and fit the model on the training data.

• Make predictions on the testing data.

• Evaluate the model's performance using MAE, R2-score, and variance score.

• Visualize the actual vs. predicted values using a line plot.

**Random Forest:**

• Preprocess the data.

• Split the data into training and testing sets.

• Build a Random Forest regression model.

• Fit the model on the training data.

• Make predictions on the testing data.

• Evaluate the model's performance using MAE, R2-score, and variance score.

• Visualize the actual vs. predicted values using a line plot.

## Visualizations

The code uses the Plotly library to create line plots for the actual vs. predicted values of each model.

**Note:** the code assumes the necessary dependencies are installed, and the dataset is in the correct format.


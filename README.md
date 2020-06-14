# Prediction_coding-Analytics_test
This repository contains the coding with the solution for the exercise in the analytics test interview. The code implements a linear regression model of the three forecast vendors from the pickle file `features.pkl` to predict the wind production of the windfarm from the pickle file `target.pkl`. Least-square and Lasso estimators are implemented to estimate the linear regression model.

## Prerequisites
The file `requirements.txt` list all the packages that are needed to run the code.

## Code structure
There are two files:
* forecast.py
* main.py

Run the file `main.py` that presents the results, which will call file `forecast.py` builds the class (`windForecast`) necessary to run the prediction model.

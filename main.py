# Import packages and class
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from forecast import windForecast # Class with the forecast techniques for this problem

# Main code
analyTestFiles = ['features', 'target'] # Files with explanatory variables (features) and response variables (target)
analyTestForcast = windForecast(analyTestFiles, 144)
# First check for non stationarity trends
# Check stationary properties in both time series - features and target
for i in range(analyTestForcast.noVendor):
    analyTestForcast.statioInfo[i] = analyTestForcast.checkStaionarity(analyTestForcast.features[i])  # Feature time series of vendor i
analyTestForcast.statioInfo[i + 1] = analyTestForcast.checkStaionarity(analyTestForcast.target[0])  # Target time series
print('Results of Dickey-Fuller Test:')
print(analyTestForcast.statioInfo)

# Second check for autocorrelation
pd.plotting.lag_plot(analyTestForcast.target[0])
plt.show()
plot_acf(analyTestForcast.target[0])
plt.ylabel('Correlation factor [0-1]')
plt.xlabel('Lab variables')
plt.show()

# Third create the linear regression model - LS estimator
# Call the training process - Split training and validation data
estimator = 'LS-estimator' # Select between LS and Lasso estimators
# estimator = 'Lasso-estimator'
analyTestForcast.trainProc(estimator)

# See the results
print('Linear regression model - ' + estimator)
print(analyTestForcast.linRegModel)
print(estimator + ' - Descritive statistics')
print(analyTestForcast.linRegModel[['R2_score', 'MAE_score', 'RMSE_score']].describe())
# Boxplot of Score indicators - R2, MAE, RMSE
analyTestForcast.linRegModel[['R2_score', 'MAE_score', 'RMSE_score']].boxplot()
# Plot the prediction of wind production
analyTestForcast.target.plot()
plt.ylabel('Energy [MWh]')
plt.xlabel('Time [30min]')
plt.show()

# See Avg weekly production
analyTarget = analyTestForcast.target.resample('60min').sum()
weekTarget = analyTarget['2019-01-14':'2019-10-27'].resample('168h').sum()
print('Average weekly production: %.2f MWh' %weekTarget[0].mean())
print('Wind production seasonality')
weekTarget.plot()
plt.ylabel('Weekly Energy [MWh]')
plt.xlabel('Time [Week]')
plt.show()

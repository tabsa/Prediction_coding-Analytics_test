# Import packages and class
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from forecast import windForecast # Class with the forecast techniques for this problem

# Main code
analyTestFiles = ['features', 'target'] # Files with explanatory variables (features) and response variables (target)
analyTestForcast = windForecast(analyTestFiles, 144)
# First check for non stationarity trends
print('Results of Dickey-Fuller Test:')
print(analyTestForcast.statioInfo)

# Second check for autocorrelation
pd.plotting.lag_plot(analyTestForcast.target[0])
plt.show()
plot_acf(analyTestForcast.target[0])
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
plt.show()

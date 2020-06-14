# Import packages
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn import metrics as sklMet
from statsmodels.tsa.stattools import adfuller

# Wind forecast class
class windForecast:
    def __init__(self, filesList, trnOmega):
        # Read the pkl-files + write them into csv-files
        self.features = self.pklFile(filesList[0], 0)
        self.target = self.pklFile(filesList[1], 0)
        # Set scalar and array parameters
        self.startDate = self.features.index[0].date() # Starting date in dt
        self.endDate = self.features.index[-1].date() # Ending date in dt
        self.noVendor = self.features.shape[1] # Number of vendors
        self.trnOmega = trnOmega  # No. days for training
        totalDays = (self.endDate - self.startDate).days + 1 # Total number of days
        self.tragOmega = totalDays - self.trnOmega  # No. days for validation
        self.timestep = int(len(self.target)/totalDays) # No. timestep
        # Set DataFrame and time-series parameters
        self.tmStep = pd.date_range('00:00', '23:30', freq='30min', tz=None).time # Generates timestamp for the 48 periods of the time series
        self.statioInfo = pd.DataFrame()  # Dataframe containing stationarity info of each time series
        # Initialize Dataframe to store the results with the Linear Regression Model
        colsName = [i for i in range(self.noVendor+1)]
        colsName.append('R2_score')
        colsName.append('MAE_score')
        colsName.append('RMSE_score')
        self.linRegModel = pd.DataFrame(np.zeros((self.timestep, self.noVendor+4)), index=self.tmStep, columns=colsName)

    def pklFile(self, filename, csvFlag): # Function to load pkl-file, and write as csv-file
        df = pkl.load(open(str(filename+'.pkl'), 'rb'))
        #
        # Cleaning process!
        # Check for NaN values
        if df.isnull().values.any() == True:
            df = df.ffill(axis=0) # Propagate the values of previous row (previous timestep t-1)
        # Convert the df.columns to float64
        df = df.astype('float64')
        # Correct the timezone - Null the timezone - Target has tz=+01:00 which shifts 1 timestep from '2019-03-31 01:00'
        df.index = df.index.tz_convert(tz=None)
        # Target dataframe - Add extra column to store the prediction
        if filename == 'target':
            df['Pred'] = np.nan

        # Write to csv the corresponding pkl-file - iff csvFlag = 1
        if csvFlag == 1:
            df.to_csv(str(filemane+'.csv'))
        return df # Return the dataframe of the respective pkl-file

    def checkStaionarity(self, test):
        # Perform Dickey-Fuller test:
        dfTest = adfuller(test, autolag='AIC')
        dfOut = pd.Series(dfTest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dfTest[4].items():
            dfOut['Critical Value (%s)' % key] = value
        # If non stationarity condition - Create function to transforming into stationarity
        return dfOut

    def trainProc(self, estimator):
        # Feature engineering part!
        # Check stationary properties in both time series - features and target
        for i in range(self.noVendor):
            self.statioInfo[i] = self.checkStaionarity(self.features[i]) # Feature time series of vendor i
        self.statioInfo[i+1] = self.checkStaionarity(self.target[0]) # Target time series
        #
        # Training process independent for each timestep t
        for t in range(self.timestep):
            # Select the corresponding datapoints of the specific time t
            auxFeatDF = self.features.at_time(self.tmStep[t]).copy() # Features df
            auxTargDF = self.target.at_time(self.tmStep[t]).copy() # Target df
            # Call the linear regression model
            idx = auxTargDF.index # Index to allocate the results to the original target Dataframe
            self.target.loc[idx] = self.linModEstimator(estimator, auxFeatDF, auxTargDF, self.tmStep[t]) # Prediction of linModel for timestep t
        print('Training process terminated!')

    def linModEstimator(self, estimator, featDF, targDF, tstep):
        # Set the dates that are used for the training and validation
        # Training data
        trnDateSet = pd.date_range(self.startDate, periods= self.trnOmega).strftime('%Y-%m-%d') # pd.datetimeindex with dates
        # Datetimeindex to separate the training and validation parts of the dataset
        npTrnFeat = featDF[trnDateSet[0]:trnDateSet[-1]].to_numpy()
        npTranTarg = targDF[trnDateSet[0]:trnDateSet[-1]].to_numpy()
        # Validation data
        tragDateSet = pd.date_range(trnDateSet[-1], periods= self.tragOmega+1).strftime('%Y-%m-%d')
        npValFeat = featDF[tragDateSet[1]:tragDateSet[-1]].to_numpy()

        # Create the lin-reg model
        if estimator == 'LS-estimator': # Least-square
            reg = LinearRegression().fit(npTrnFeat, npTranTarg[:, 0])
        elif estimator == 'Lasso-estimator': # Lasso
            reg = Lasso(alpha=0.1).fit(npTrnFeat, npTranTarg[:, 0])
        # Predict the future time lag
        targDF['Pred'][tragDateSet[1]:tragDateSet[-1]] = reg.predict(npValFeat)
        # Score the lin-reg model for timestep t
        regR2 = sklMet.r2_score(targDF[0][tragDateSet[1]:tragDateSet[-1]], targDF['Pred'][tragDateSet[1]:tragDateSet[-1]])
        regMAE = sklMet.mean_absolute_error(targDF[0][tragDateSet[1]:tragDateSet[-1]], targDF['Pred'][tragDateSet[1]:tragDateSet[-1]])
        regRMSE = np.sqrt(sklMet.mean_squared_error(targDF[0][tragDateSet[1]:tragDateSet[-1]], targDF['Pred'][tragDateSet[1]:tragDateSet[-1]]))

        # Final result to be allocated to the self.target dataframe
        self.linRegModel.loc[tstep] = np.concatenate(([reg.intercept_], reg.coef_, [regR2, regMAE, regRMSE]), axis=0)
        return targDF

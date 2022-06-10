'''
Analyze absenteeism datasets and identify the features that contribute most to absenteeism.

Classes:
    CustomScaler: Custom scaler class to scale the data.
    AbsenteeismModel: Class to clean the dataset and create a model to predict absenteeism.
'''

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    '''Custom scaler class to scale the data.'''

    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self

    def transform(self, X):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

class AbsenteeismModel():
    '''Class to clean the dataset and create a model to predict absenteeism.'''

    def __init__(self, model_file, scaler_file):
        '''read the 'model' and 'scaler' files and set them as attributes'''
        with open('model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    def load_and_clean_data(self, data_file):
        '''Takes a *.csv data file and returns a cleaned version for use by the model'''
        #import the data
        df = pd.read_csv(data_file, delimiter=',')
        # store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        # drop the 'ID' column
        df.drop(columns='ID', inplace=True)
        # add the absenteeism column that will be predicted with NaN values for now
        df['Absenteeism Time in Hours'] = 'NaN'

        # create a separate dataframe with dummy values for all available reasons
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)

        # split reason_columns into the 4 reason categories
        reason_type_1 = reason_columns.loc[:, :14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        reason_type_1.fillna(0, inplace=True)
        reason_type_2.fillna(0, inplace=True)
        reason_type_3.fillna(0, inplace=True)
        reason_type_4.fillna(0, inplace=True)

        reason_type_1 = reason_type_1.astype(int)
        reason_type_2 = reason_type_2.astype(int)
        reason_type_3 = reason_type_3.astype(int)
        reason_type_4 = reason_type_4.astype(int)

        # to avoid multicollinearity, drop the 'Reason for Absence' column from df
        df.drop(columns='Reason for Absence', inplace=True)

        # concatenate df and the 4 reason categories
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        # assign names to the 4 reason category columns
        column_names = df.columns.values.tolist()[:-4] + ['Reason_' + str(i) for i in range(1, 5)]
        df.columns = column_names

        # reorder the columns in df
        column_names_reordered = ['Reason_' + str(i) for i in range(1, 5)] + df.columns.values.tolist()[:-4]
        df = df[column_names_reordered]

        # conver the 'Date' column into a timestamp datatype
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # create feature called 'Month Value' with the month retrieved from the 'Date' column
        df['Month Value'] = df.Date.apply(lambda x: x.month)

        # create feature called 'Dat of the Week' with the day of the week retrieved from the 'Date' column
        df['Day of the Week'] = df.Date.apply(lambda x: x.weekday())

        # drop the 'Date' column from df
        df.drop(columns='Date', inplace=True)

        # reorder the columns in df
        df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                 'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                 'Daily Work Load Average', 'Body Mass Index', 'Education',
                 'Children', 'Pets', 'Absenteeism Time in Hours']]

        # map the values in the 'Education' column to be binary
        df.Education = df.Education.map({1:0, 2:1, 3:1, 4:1})

        # replace the NaN values
        df = df.fillna(value=0)

        # drop the original absenteeism time column
        df.drop(columns='Absenteeism Time in Hours', inplace=True)

        # drop the features that we don't need
        df.drop(columns=['Daily Work Load Average', 'Distance to Work'], inplace=True)

        # copy the precessed data to a new dataframe
        self.preprocessed_data = df.copy()

        # scale the data
        self.data = self.scaler.transform(df)

    def predicted_probability(self):
        '''Outputs the predicted probability of each row in the dataframe'''
        if self.data is not None:
            pred = self.reg.predicted_proba(self.data)[:, 1]
            return pred
        return None

    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        return None

    def predicted_outputs(self):
        '''Predict outputs with associated probabilities and add columns with these values to the dataframe'''
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data
        return None

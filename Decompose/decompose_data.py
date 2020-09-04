import numpy as np
import pandas as pd
from Decompose.fit_trend_methods import trend_method
from matplotlib import pyplot as plt

# This class decompose the temporal series and return it without trend and heterocedasticity
# After It predicts the trend in the testing part and sum to the prediction.
# The class imports the trend_method class that contains the definition of the different trend fitting models.

class Decompose(object):
    def __init__(self, x_train_features, data_train, log_transform, trend_method_name, parameters_trend, mode_dataset):
        self.poly_features = None
        self.line_reg = None
        self.x_train_features = x_train_features
        self.data_train = data_train

        self.trend_method_name = trend_method_name
        self.log_transform = log_transform
        self.model_fit_trend = None
        self.parameters_trend = parameters_trend
        self.mode_dataset = mode_dataset

    # ElIMINATES THE TREND AND HETEROCEDASTICITY : return the data_train without these components
    def descompose_train_data(self):
        if self.log_transform and self.mode_dataset == 'diff':
            # PREPROCESSING DATA: heterocedasticity (only available for polynomic fitting)
            data_train_log = np.log(self.data_train)
            data_train_log[data_train_log == -np.inf] = 0    #It will put a 0 in the NaN samples caused by np.log
            self.data_train = data_train_log

        # PREPROCESSING DATA: returng the data_train without trend
        data_train_without_trend = self.fit_trend_apply()
        return data_train_without_trend

    #CALLS THE FIT_TREND_METHODS FILE:
    # - define the trend_method object, fit the trend of the curve and return the data_train without trend
    # - generates a plot to check if the the trend of the curve is well-fitted.
    def fit_trend_apply(self): ### Calls the fit_trend_methods file
        self.model_fit_trend = trend_method(self.x_train_features, self.data_train, self.trend_method_name,
                                            self.parameters_trend, self.mode_dataset)
        self.model_fit_trend.fit()
        trend_train = self.model_fit_trend.predict(self.x_train_features)
        trend_train = pd.DataFrame(trend_train)
        data_without_trend = self.data_train - trend_train.values

        #CHECKING PLOT
        fig, ax = plt.subplots(figsize=(9,5))
        ax.plot(self.x_train_features[:, 0].squeeze(), trend_train, label='trend_train')
        ax.plot(self.x_train_features[:, 0].squeeze(), self.data_train, label = 'data_train')
        ax.plot(self.x_train_features[:, 0].squeeze(), data_without_trend, label = 'data_train_without_trend')
        ax.legend()
        ax.set_title('FIT_TREND_CHECK')
        return data_without_trend

    #COMPOSE THE PREDICTION ADDING ITS TREND AND HETEROCEDASTICITY:
    # It needs the prediction of the ML model and the x_features (days of the prediction)
    def predict_compose(self, x_features, prediction_without_trend):
        predict_trend = self.model_fit_trend.predict(x_features)
        predict_trend = pd.DataFrame(predict_trend)
        if self.log_transform:  #log(Series) have been applied

            final_prediction = np.exp(prediction_without_trend.reshape(-1,1) + predict_trend.values)
            final_prediction = np.nan_to_num(final_prediction)

        else:
            final_prediction = prediction_without_trend.reshape(-1,1) + predict_trend.values
            final_prediction = np.nan_to_num(final_prediction)

        return final_prediction


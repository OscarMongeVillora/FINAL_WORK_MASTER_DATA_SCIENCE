from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np

class SARIMA_model(object):
    def __init__(self, list_hyperparam_model, param_estacionality):
        self.p, self.d, self.q, self.P, self.D, self.Q = list_hyperparam_model
        self.results = None
        self.param_estacionality = param_estacionality

    def fit(self, x_features, data_train):
        #data_train un dataframe con indice_temporal_ojo
        #data_train.index = pd.to_datetime(data_train.index)
        model = SARIMAX(data_train.squeeze(), order=(self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, self.param_estacionality),
                        enforce_stationarity=False, enforce_invertibility=False)
        #print(data_train.squeeze())
        self.results = model.fit(maxiter=200, disp=0, method='nm')


    def predict(self, x_features):

        predictions = self.results.get_forecast(steps=x_features.shape[0])
        pred = predictions.predicted_mean
        pred = pd.DataFrame(pred)
        pred.fillna(0, inplace = True)
        pred = np.array(pred)

        return pred


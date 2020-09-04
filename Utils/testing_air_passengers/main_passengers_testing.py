from Bayes_Optimization.bayes_optim import Bayesian_Opt
import numpy as np
import pandas as pd
from Plots_and_error.results import evaluate

def main_passenger_dummy():
    name_dataset = 'dummy'  # Which dataset is
    mode_dataset = 'diff'  # We obtain the dataset in acumulated version, we modeled his acum. temporal series without tendency and obtain the acum. prediction
    method_trend = 'polynomic'  # Which method of fitting the trend: polynomic, logistic, gompertz
    predict_intervals = False
    log_transform = True
    parameters_trend = 1
    data = pd.read_csv("AirPassengers.csv", parse_dates=['Month'], index_col='Month', header=0)
    print(data.shape[0])

    TEST_SIZE = 24

    data_train, data_test = data.iloc[:-TEST_SIZE], data.iloc[-TEST_SIZE:]
    x_train = np.array(range(data_train.shape[0]))
    x_test = np.array(range(data_train.shape[0], data.shape[0]))

    x_train_features = np.hstack((x_train.reshape(-1,1), np.array(list(range(12)) * 10).reshape(-1,1)))
    x_test_features = np.hstack((x_test.reshape(-1,1), np.array(list(range(12)) * 2).reshape(-1,1)))

    model = Bayesian_Opt(x_train_features,
                         data_train,
                         n_sample_subset=30,  # Need to be multiple of the season_length
                         model_choose='SARIMA',
                         predict_intervals=predict_intervals,
                         loss_function=0,  # This is the loss function for the obj function of bayesian optimization
                         log_transform=log_transform,
                         fit_trend_method=method_trend,
                         parameters_trend=parameters_trend,
                         mode_dataset=mode_dataset,
                         timesplitmethod=0,
                         percentage_validation=0.2,
                         overlap=0.3)
    param, eval = model.run(max_iter=30)
    # print(eval)
    print(param)  # They aren't the last param corresponding to the last iteration. They correspond to the best rsme.
    model.fit()
    prediction_train = model.predict(x_train_features)
    prediction_test = model.predict(x_test_features)
    store_data = dict()
    store_data['air'] = [x_train_features[:, 0], data_train, prediction_train,
                    x_test_features[:, 0], data_test, prediction_test]

    evaluate(store_data, name_dataset = name_dataset, mode_dataset = mode_dataset, metrics_print = True, train_predict_plot = False)
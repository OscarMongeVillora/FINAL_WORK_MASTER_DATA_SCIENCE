from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np

#This module prints and returns the prediction error for a metric chosen

def metrics_evaluation(data_train, data_test, prediction_train, prediction_test, list_metrics=None, mode ='Full',
                       predict_intervals = False):

    # This is the predetermined list of metrics chosen
    if list_metrics is None:
        #set_metrics = {'mae', 'rmse', 'rmsle'}
        set_metrics = {'mae', 'rmse'}
    else:
        set_metrics = set(list_metrics)

    if  predict_intervals == True:  #We only comput the error for the mid prediction, not for the bounds
        prediction_train = prediction_train[0]
        prediction_test = prediction_test[0]

    #Calculate metrics errors:
    mae_train = mae(data_train, prediction_train)
    mae_test = mae(data_test, prediction_test)
    rmse_train = np.sqrt(mse(data_train, prediction_train))
    rmse_test = np.sqrt(mse(data_test, prediction_test))
    #rmsle_train = np.sqrt(mse(np.log(data_train + 1), np.log(prediction_train + 1)))
    #rmsle_test =  np.sqrt(mse(np.log(data_test + 1), np.log(prediction_test + 1)))
    rmsle_train, rmsle_test = [None, None]
    mape_train = np.mean(np.abs((data_train - prediction_train) / data_train))
    mape_test = np.mean(np.abs((data_test - prediction_test) / data_test))

    ## Create return test error dictionary for each metrics
    metric_return_dict = {}
    if 'mae' in set_metrics:
        metric_return_dict['mae'] = mae_test
    if 'rmse' in set_metrics:
        metric_return_dict['rmse'] = rmse_test
    if 'mape' in set_metrics:
        metric_return_dict['mape'] = mape_test


    ######## 2 MODES OF PRINTING ERROR : Both return the error metrics values.
    #'Full' mode print the error for the different metrics
    #'Inplot' mode incorporate the error in the plots.

    if mode == 'Full':
        if 'mae' in set_metrics:
            print('\n','\n','mae using Bayes Optimization CV (train): ',mae_train)
        if 'rmse' in set_metrics:
            print('rmse using Bayes Optimization CV (train): ', rmse_train)
        if 'rmsle' in set_metrics:
            print('rmsle using Bayes Optimization CV (train):', rmsle_train)  #Avoid Outlayers
        if 'mape' in set_metrics:
            print('mape using Bayes Optimization CV (train):', mape_train)

        if 'mae' in set_metrics:
            print('\n', '\n', 'mae using Bayes Optimization CV (test): ', mae_test)
        if 'rmse' in set_metrics:
            print('rmse using Bayes Optimization CV (test): ', rmse_test)
        if 'rmsle' in set_metrics:
            print('rmsle using Bayes Optimization CV (test):', rmsle_test, '\n')  # Avoid Outlayers
        if 'mape' in set_metrics:
            print('mape using Bayes Optimization CV (train):', mape_test)

        return metric_return_dict

    if mode == 'Inplot':
        #Here the metrics are always the same (are fixed)
        str1 = str('MAE: {:.3f}, RMSE: {:.3f} (TRAIN)'.format( mae_train, rmse_train))
        str2 = str('MAE: {:.3f}, RMSE: {:.3f} (TEST)'.format( mae_test, rmse_test))
        #str1 = str('MAE: {:.3f}, RMSE: {:.3f}, RMSLE: {:.3f} (TRAIN)'.format(mae_train, rmse_train, rmsle_train))
        #str2 = str('MAE: {:.3f}, RMSE: {:.3f}, RMSLE: {:.3f} (TEST)'.format(mae_test, rmse_test, rmsle_test))

        return str1, str2, metric_return_dict


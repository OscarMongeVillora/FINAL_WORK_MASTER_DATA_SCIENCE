from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Plots_and_error.evaluation_metrics import metrics_evaluation

#This module carries out the plots of the train and test part of the temporal series with the prediction on test.
# It has 3 modes of plotting : visualize the different communities predictions in various perspectives
# It returns the RMSE error and the vectors of the predictions too (the predictions are returned in the last mode).


#'plot_mode' = 0  PLOTS THE PREDICTION OF EACH REGION ONE AFTER ONE
def evaluate(store_data, name_dataset, mode_dataset,  metrics_print,  plot_print,  predict_intervals = False,
             list_notdefault_metrics = None, metrics_return = 'rmse', train_predict_plot = True):

    metrics_error_comunity = {}
    for com in store_data.keys():

        #Extract the features, the Ground Truth and the predictions for each community series of the input store_data
        x_train, data_train, prediction_train, x_test, data_test, prediction_test = store_data[com]

        #Call the metric_evaluation module that prints the evaluation error for the predictions.
        if metrics_print:
            error_test_dict = metrics_evaluation(data_train.values, data_test.values, prediction_train,
                                                 prediction_test, list_metrics = list_notdefault_metrics,
                                                 predict_intervals = predict_intervals)
            if metrics_return not in error_test_dict.keys():
                raise ValueError("The metrics you want to return is not in the default list, so you need to add it"
                                 "in the list_not_default_metrics")

            # Select the error test for the metrics_return choosen and keep it in a dictionary for each community
            error_test = error_test_dict[metrics_return]
            metrics_error_comunity[com] = error_test

        #Plot the predictions for each community
        if plot_print:
            fig, ax = plt.subplots(figsize=(15,5))
            ax.plot(x_train, data_train, label='train')
            ax.plot(x_test, data_test, label='test', lw=3)
            if predict_intervals == False:
                if train_predict_plot == True:
                    ax.plot(x_train, prediction_train, label='predict_train_BAYES_CV'  )
                ax.plot(x_test, prediction_test, label='predict_test_BAYES_CV_', lw=2)
            if predict_intervals == True:
                ax.plot(x_train, prediction_train[0], label='predict_train_BAYES_CV')
                ax.plot(x_test, prediction_test[0], label='predict_test_BAYES_CV_', lw=2)
                #ax.plot(x_test, prediction_test[1],'--', lw=2)
                #ax.plot(x_test, prediction_test[2],'--', lw=2)
                ax.fill_between(x_test, prediction_test[2].squeeze(), prediction_test[1].squeeze(), color= 'blue',
                                linestyle="--", alpha = 0.2)
            ax.set_title(title_def(name_dataset, mode_dataset) + ' in {}'.format(com))
            ax.legend()
            ax.grid()
            plt.show()

        return metrics_error_comunity

#'plot_mode' =1  PLOTS A UNIQUE PLOT WITH AN ARRANGE OF SUBFIGURES, EACH ONE FOR SHOWING THE PREDICTIONS OF EACH REGION
def plot_comunities(store_data, name_dataset, mode_dataset, metrics_print, plot_print, predict_intervals = False,
                    metrics_return = 'rmse', train_predict_plot = True):

    #Create the arrange of subfigure depending the number of communities selected
    elem = len(store_data.keys())
    matrix_k = int(np.sqrt(elem))
    rest = elem - matrix_k * matrix_k
    columns = matrix_k
    rows = matrix_k
    if rest != 0:
        if rest <= matrix_k:
            rows = rows + 1
        elif rest <= matrix_k * 2:
            rows = rows + 2
        elif rest <= matrix_k * 3:
            rows = rows + 3

    metrics_error_comunity = {}

    #Plots the figure
    fig = plt.figure(figsize=(15, 9))
    for i_com, com in enumerate(store_data.keys()):

        # Extract the features, the Ground Truth and the predictions for each community series of the input store_data
        # Calculate the error for the predictions of each community
        x_train, data_train, prediction_train, x_test, data_test, prediction_test = store_data[com]
        str1, str2, error_test_dict = metrics_evaluation(data_train, data_test, prediction_train, prediction_test,
                                                            mode='Inplot', predict_intervals = predict_intervals)
        if metrics_return not in error_test_dict.keys():
            raise ValueError("The metrics you want to return is not in the default list, so you need to add it "
                             "in the list_not_default_metrics")

        # Select the error test for the metrics_return choosen and keep it in a dictionary for each community
        error_test = error_test_dict[metrics_return]
        metrics_error_comunity[com] = error_test

        #Plot each subfigure of the plot
        if plot_print:
            ax = fig.add_subplot(rows, columns, i_com + 1)
            ax.plot(x_train, data_train, label='train')
            ax.plot(x_test, data_test, label='test', lw=3)
            if predict_intervals == False:
                if train_predict_plot == True:
                    ax.plot(x_train, prediction_train, label='predict_train_BAYES_CV')
                ax.plot(x_test, prediction_test, label='predict_test_BAYES_CV_', lw=2)
            if predict_intervals == True:
                ax.plot(x_train, prediction_train[0], label='predict_train_BAYES_CV')
                ax.plot(x_test, prediction_test[0], label='predict_test_BAYES_CV_', lw=2)
                #ax.plot(x_test, prediction_test[1], lw=2)
                #ax.plot(x_test, prediction_test[2], lw=2)
                ax.fill_between(x_test, prediction_test[2].squeeze(), prediction_test[1].squeeze(), color='blue',
                                linestyle="--", alpha=0.2)
            if i_com == 0:
                ax.legend()
            ax.grid()

            if metrics_print:
                ax.text((x_test[-1]) * 3/4, np.max(data_train.values.squeeze())/2,
                        str1 + '\n' + str2, horizontalalignment='center',  verticalalignment='center')
            ax.set_title(title_def(name_dataset, mode_dataset) + ' in {}'.format(com))

    plt.show()
    return metrics_error_comunity


#'plot_mode' =2  PLOTS A UNIQUE PLOT WITH THE SUM OF THE PREDICTIONS AND DATA FOR THE COMMUNITIES SELECTED
def global_prediction_plot(store_data, name_dataset, mode_dataset, metrics_print, plot_print, predict_intervals =False,
                           list_notdefault_metrics = None, metrics_return = 'rmse', train_predict_plot = True):

    x_train_global = [x for x in store_data.values()][0][0]
    x_test_global = [x for x in store_data.values()][0][3]

    #Initialize the global arrays (predicting_intervals = False)
    prediction_train_global = np.zeros(len(x_train_global)).reshape(1,-1)
    prediction_test_global = np.zeros(len(x_test_global)).reshape(1,-1)
    data_train_global = np.zeros(len(x_train_global)).reshape(1,-1)
    data_test_global = np.zeros(len(x_test_global)).reshape(1,-1)

    #Initialize the global arrays (predicting_intervals = True)
    prediction_train_global_down = np.zeros(len(x_train_global)).reshape(1,-1)
    prediction_train_global_up =  np.zeros(len(x_train_global)).reshape(1,-1)
    prediction_test_global_down = np.zeros(len(x_test_global)).reshape(1,-1)
    prediction_test_global_up = np.zeros(len(x_test_global)).reshape(1,-1)

    #Sum the Ground Truth and Predictions for all the communities selected.
    for data in store_data.values():
        data_train_global += np.array(data[1].values.reshape(1, -1))
        data_test_global += np.array(data[4].values.reshape(1, -1))

        if predict_intervals == False:
            prediction_train_global += np.array(data[2].reshape(1,-1))
            prediction_test_global += np.array(data[5].reshape(1,-1))
        if predict_intervals == True:
            prediction_train_global += np.array(data[2][0].reshape(1, -1))
            prediction_train_global_down += np.array(data[2][1].reshape(1, -1))
            prediction_train_global_up += np.array(data[2][2].reshape(1, -1))

            prediction_test_global += np.array(data[5][0].reshape(1, -1))
            prediction_test_global_down += np.array(data[5][1].reshape(1, -1))
            prediction_test_global_up += np.array(data[5][2].reshape(1, -1))

    #Generate the output global array of predictions
    data_out = [data_train_global, data_test_global, prediction_train_global, prediction_test_global]

    error_test = -99999

    # Calculate the global error for the predictions of all the communities
    if metrics_print:
        error_test_dict = metrics_evaluation(data_train_global, data_test_global, prediction_train_global,
                                             prediction_test_global,list_metrics = list_notdefault_metrics)
        if metrics_return not in error_test_dict.keys():
            raise ValueError("The metrics you want to return is not in the default list, so you need to add it "
                             "in the list_not_default_metrics")
        error_test = error_test_dict[metrics_return]

    #Plots the figure
    if plot_print:
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(x_train_global, data_train_global.squeeze(), label='train')
        ax.plot(x_test_global, data_test_global.squeeze(), label='test', lw=3)
        if train_predict_plot == True:
            ax.plot(x_train_global, prediction_train_global.squeeze(), label='predict_train_BAYES_CV')
        ax.plot(x_test_global[1:], prediction_test_global.squeeze()[1:], label='predict_test_BAYES_CV_', lw=2)

        if predict_intervals == True:
            #ax.plot(x_test_global, prediction_test_global_down.squeeze(), '--', lw=2)
            #ax.plot(x_test_global, prediction_test_global_up.squeeze(), '--', lw=2)
            ax.fill_between(x_test_global, prediction_test_global_up.squeeze(), prediction_test_global_down.squeeze(), color='blue',
                            linestyle="--", alpha=0.2)

        ax.set_title(title_def(name_dataset, mode_dataset))
        ax.legend()
        ax.grid()
        plt.show()

    return error_test, data_out





def title_def(name_dataset, mode_dataset):
    title = None
    if (name_dataset == 'deaths_old') | (name_dataset == 'deaths_update'):
        if mode_dataset == 'diff':
            title = 'Prediction of daily deaths'
        if mode_dataset == 'cum':
            title = 'Prediction of acumulated deaths'

    if (name_dataset == 'casos_update') | (name_dataset == 'national'):
        if mode_dataset == 'diff':
            title = 'Prediction of daily cases'
        if mode_dataset == 'cum':
            title = 'Prediction of acumulated cases'
    else:
            title = name_dataset

    return title







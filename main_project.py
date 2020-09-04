from Bayes_Optimization.bayes_optim import Bayesian_Opt
import numpy as np
import pandas as pd
from Plots_and_error.results import evaluate, plot_comunities, global_prediction_plot
from Datasets.datasets_prepare import import_datasets
from Utils.feature_engineering import stack_features
from Decompose.parameters_adjusted_datasets import load_parameters_adjusted
import sys
from config_generator import config_generator
import configparser

# The module contains the main instructions of the project. It imports the configuration parameters, prepares the data
# of the time series and run the Bayes Optimization module to obtain an accurate prediction of the series
# The most important function of this module that runs all the project is main_project().

#VERIFIES THE COMPATIBILITY OF THE CONFIGURATION PARAMETERS CHOSEN. 2 functions:
# 1- Avoid bad configurations of the run
# 2- Obtain the log_transform value and the train_predict_plot value (adjust these internal parameters)
def check_programmed_ways(mode_dataset, method_trend, ML_model_choose):
    train_predict_plot = True

    if (mode_dataset == 'cum') and (method_trend == 'polynomic'):
        print('\n','It is not a programmed option using a acumulated dataset '
                   'with a polynomic trend.','\n','\n','Put again the parameters')
        sys.exit()

    if method_trend == 'polynomic':  #We only apply log(Series) if after we fit the trend with a polynomic curve
        log_transform = True
    else:
        log_transform = False

    if (ML_model_choose == 'SARIMA') or (ML_model_choose == 'NN'):
        train_predict_plot = False

    return log_transform, train_predict_plot

#PREPARE THE DATA FOR ONE COMUNITY SELECTED AND RETURN IT PROCESSED IN A LIST
def data_prepare(data, Comunity:str, size_train: int, name_dataset: str):
    if data.shape[0] == 1:  #For this Datasets that only talk about one region as a Nation for example -- 1 row
        dat_com = data.T
    else:
        dat_com = pd.DataFrame(data.loc[Comunity])
    size_dat = len(dat_com)
    print('Total sample size of the dataset prepared: ',size_dat)

    #Split into train and test
    SIZE_TRAIN = size_train
    data_train, data_test = dat_com.iloc[:SIZE_TRAIN], dat_com.iloc[SIZE_TRAIN:]
    x_train, x_test = np.array(range(SIZE_TRAIN)), np.array(range(SIZE_TRAIN, dat_com.shape[0]))

    #Feature Engineering (General)
    season_length = 7
    intervals = int(np.ceil(size_dat / season_length))
    vec = np.tile(list(range(season_length)), intervals)
    x_train_week = np.hstack((x_train.reshape(-1, 1), vec[:SIZE_TRAIN].reshape(-1, 1)))
    x_test_week = np.hstack((x_test.reshape(-1, 1), vec[SIZE_TRAIN : SIZE_TRAIN + len(x_test)].reshape(-1, 1)))

    #Specific Engineering feature: Add more features for 'casos_update' dataset
    x_train_week, x_test_week = stack_features(name_dataset, size_dat, SIZE_TRAIN, x_train_week, x_test_week,
                                               feature_wave= 0, feature_sign = 0)

    return [x_train_week, data_train, x_test_week, data_test]

#READS THE CONFIGURATION VALUES OF "config.ini" AND KEEP THEM IN A DICTIONARY "conf".
def config_function():
    config = configparser.ConfigParser()
    config.read('config.ini')
    conf = {}
    conf[0] = config['DATASET_CHOOSEN']['name_dataset']
    conf[1] = config['DATASET_CHOOSEN']['mode_dataset']
    conf[2] = config['TREND_FIT']['method_trend']
    conf[3] = config['ML_METHOD']['ml_model_choose']
    conf[4] = config['ML_METHOD'].getboolean('predict_intervals')
    conf[6] = int(config['DATASET_STRUCTURE']['size_train'])
    conf[7] = int(config['CV_BAYES']['n_sample_subset'])
    conf[8] = int(config['CV_BAYES']['loss_function'])
    conf[9] = int(config['TIME_SPLIT_CV']['timesplitmethod'])
    conf[10] = float(config['TIME_SPLIT_CV']['percentage_validation'])
    conf[11] = float(config['TIME_SPLIT_CV']['overlap'])
    conf[12] = int(config['RUN_BAYES_OPT_ITERATION']['max_iter'])
    conf[13] = int(config['PLOT_CONFIGURATION']['plot_mode'])
    conf[14] = config['PLOT_CONFIGURATION'].getboolean('metrics_print')
    conf[15] = config['PLOT_CONFIGURATION'].getboolean('plot_print')
    conf[16] = config['RUN_BAYES_OPT_ITERATION'].getboolean('bayes_tuning_on')
    print(conf)
    return conf

#MAIN FUNCTION OF THE MODULE: RUN THE PROGRAM
def main_project(config_object = config_generator()):

    #Import the list of communities and overwrite the configuration values by those selected on config_generator.py
    # If activate = False, read the existent config.ini instead of overwritting
    list_comunities = config_object.config_replace_ini(activate = True)
    conf = config_function()  #Keep the dictionary with all the configured parameters

    #Verifies a good set of parametes is choosen, import the dataset choosen of the import_datasets module and
    #load the adjusted and fixed parameters for the fitting trend methods
    log_transform, train_predict_plot = check_programmed_ways(conf[1], conf[2], conf[3])
    data_clean= import_datasets(name = conf[0], mode = conf[1])
    parameters_trend = load_parameters_adjusted(conf[0], conf[2], data_clean)
    print('Adjusted parameters for fitting the trend: ',parameters_trend)

    #If True select all communities of the dataset
    if list_comunities == 'all':
        list_comunities = parameters_trend.keys()

    store_data = dict()
    for com in list_comunities:

        #Prepare and process the dataframe for one community or region of the dataset
        x_train_features, data_train, x_test_features, data_test = data_prepare(data_clean, com, size_train= conf[6],
                                                                                name_dataset = conf[0])
        #Inicialize the Bayesian_Opt Object with all the configuration parameters chosen
        model = Bayesian_Opt(x_train_features,
                         data_train,
                         n_sample_subset = conf[7],
                         model_choose = conf[3],
                         predict_intervals = conf[4],
                         loss_function = conf[8],
                         log_transform = log_transform,
                         fit_trend_method = conf[2],
                         parameters_trend = parameters_trend[com],
                         mode_dataset = conf[1],
                         timesplitmethod=conf[9],
                         percentage_validation=conf[10],
                         overlap = conf[11],
                         bayes_tuning_on = conf[16])

        #Run the Bayesian Optimization
        param, eval = model.run(max_iter = conf[12])
        print('Optimal parameters of the ML model', param)
        #They aren't the last parameters corresponding to the last iteration. They correspond to the best rsme.

        #Fit the best model and predicts
        model.fit()
        prediction_train = model.predict(x_train_features)
        prediction_test = model.predict(x_test_features)
        store_data[com] = [x_train_features[:,0], data_train, prediction_train,
                            x_test_features[:,0], data_test, prediction_test]


    ## Plot and Error functions (1 after 1, all together in different subfigures, all together added in same figure).
    # We obtain the error test (default = rmse) and predictions (for the last mode)
    if conf[13] == 0:
        error_test = evaluate(store_data, conf[0], conf[1], metrics_print = conf[14], plot_print = conf[15],
                              predict_intervals = conf[4], train_predict_plot = train_predict_plot)
    if conf[13] == 1:
        error_test = plot_comunities(store_data, conf[0], conf[1], metrics_print = conf[14], plot_print = conf[15],
                                     predict_intervals = conf[4], train_predict_plot = train_predict_plot)
    if conf[13] == 2:
        error_test, data_out = global_prediction_plot(store_data, conf[0], conf[1], metrics_print= conf[14],
                                                      plot_print = conf[15], predict_intervals = conf[4],
                                                      train_predict_plot = train_predict_plot)
        return error_test, data_out


    print('finish')
    return error_test
import configparser

#This class allows to configure all the parameters of the project. The main_project.py module receives an object of
# this class as an input and call the method config_replace_ini of the object to overwritte the values of the config.ini

class config_generator(object):
    def __init__(self):
        #DEFINE MANUALLY THE CONFIGURATION PREFERRED
        self.parm_opt = {'name_dataset': 'deaths_update', # DATASET_CHOOSEN
                        'mode_dataset': 'diff',           # MODE OF THE DATASET: 'cum': cumulative, 'diff': differential
                        'method_trend' : 'gompertz',      #TREND FIT METHOD
                        'ML_model_choose': 'SARIMA',      #MACHINE LEARNING METHOD
                        'predict_intervals': False,       #PREDICT INTERVALS ACTIVATED
                        'size_train': 56,                 # SIZE TRAIN PART
                        'list_comunities':['Cataluña', 'Madrid'], ## LIST OF COMUNITIES SELECTED
                        'all_comunities_bool': False,     ## SELECT ALL COMUNITIES OF DATASET
                        'n_sample_subset': 32,            # CV BAYES: Number samples CV subsets.
                        'loss_function':0,                # CV BAYES: Type of loss function in Bayes Opt: 0:RMSE, 1:R^2
                        'timesplitmethod':1,              # TIME SPLIT METHOD FOR CV
                        'percentage_validation':0.3,      # PERCENTAGE OF VALIDATION SUBSET IN CV
                        'overlap':0.1,                    # OVERLAP IN METHOD 1 OF TIMESPLITTING
                        'max_iter': 20,                   # NUMBER OF ITERATION OF BAYES METHOD
                        'plot_mode': 2,                   # MODE OF PLOTTING (1vs1, together...)
                        'metrics_print' : True,           # PREDICTION ERROR SHOWN
                        'plot_print': True,               # PREDICTION PLOT SHOWN
                        'bayes_tuning_on': True}          # BAYES TUNNING ACTIVATED

        # We specified the next values available for some relevant parameters:

        ## 'name_dataset': 'deaths_old', 'deaths_update', 'casos_update', 'worlwide'
        # 'method_trend' : 'polynomic', 'gompertz', 'logistic', 'sir_model'.
        # 'ML_model_choose': 'xgb': XGBoost, 'rf': Random Forest, 'SARIMA': SARIMA Model, 'NN': Neuronal Network
        # 'timesplitmethod': 0 (Time Series Split), 1 (Blocking Time Series Split)
        # 'plot_mode': 0 (1 vs 1 regions), 1 (all regions in subplots of the same figure), 2 (all regions sum together)
        # 'size_train'. Recommended for each dataset: 56 samples (deaths_old), 64 (deaths_update),
                      # 150 (casos_update), 200 (worlwide)

    # DEFINE A CONFIGURATION TEMPLATE TO MAKE SIMULATIONS (after initializing the object config_generator)
    def import_options_template(self, name_dataset, mode_dataset, method_trend, ML_model_choose, predict_intervals,
                                size_train, list_comunities, all_comunities_bool, n_sample_subset, loss_function,
                                timesplitmethod,percentage_validation, overlap, max_iter, plot_mode, metrics_print,
                                plot_print, bayes_tuning_on):

        self.parm_opt['name_dataset'] = name_dataset
        self.parm_opt['mode_dataset'] = mode_dataset
        self.parm_opt['method_trend'] = method_trend
        self.parm_opt['ML_model_choose'] = ML_model_choose
        self.parm_opt['predict_intervals'] = predict_intervals
        self.parm_opt['size_train'] = size_train
        self.parm_opt['list_comunities'] = list_comunities
        self.parm_opt['all_comunities_bool'] = all_comunities_bool
        self.parm_opt['n_sample_subset'] = n_sample_subset
        self.parm_opt['loss_function'] = loss_function
        self.parm_opt['timesplitmethod'] = timesplitmethod
        self.parm_opt['percentage_validation'] = percentage_validation
        self.parm_opt['overlap'] = overlap
        self.parm_opt['max_iter'] = max_iter
        self.parm_opt['plot_mode'] = plot_mode
        self.parm_opt['metrics_print'] = metrics_print
        self.parm_opt['plot_print'] = plot_print
        self.parm_opt['bayes_tuning_on'] = bayes_tuning_on

    # CHANGE A VALUE FOR A SPECIFIC PARAMETER
    def modify_options(self, **kwargs):
        for key, value in kwargs.items():
            self.parm_opt[key] = value

    # CREATE THE PARSER CONFIGS OPTIONS (that will create the config.ini)
    def config_replace_ini(self, activate = False):
        config = configparser.ConfigParser()
        config['DATASET_CHOOSEN'] = {'name_dataset' : self.parm_opt['name_dataset'],
                                     'mode_dataset' : self.parm_opt['mode_dataset']}
        config['TREND_FIT'] = {'method_trend' : self.parm_opt['method_trend']}
        config['ML_METHOD'] = {'ML_model_choose' : self.parm_opt['ML_model_choose'] ,
                               'predict_intervals' : self.parm_opt['predict_intervals']}
        config['DATASET_STRUCTURE'] = {'size_train': self.parm_opt['size_train']}
        config['CV_BAYES'] = {'n_sample_subset' : self.parm_opt['n_sample_subset'] ,
                              'loss_function' : self.parm_opt['loss_function']}
        config['TIME_SPLIT_CV'] = {'timesplitmethod' :self.parm_opt['timesplitmethod'],
                                   'percentage_validation':self.parm_opt['percentage_validation'],
                                   'overlap': self.parm_opt['overlap']}
        config['RUN_BAYES_OPT_ITERATION'] = {'max_iter' : self.parm_opt['max_iter'],
                                             'bayes_tuning_on': self.parm_opt['bayes_tuning_on'] }
        config['PLOT_CONFIGURATION'] = {'plot_mode' : self.parm_opt['plot_mode'],
                                        'metrics_print': self.parm_opt['metrics_print'],
                                        'plot_print': self.parm_opt['plot_print']}

        #OVERWRITE THE CONFIG.INI FILE
        if activate == True:
            with open('config.ini', 'w') as fileconfig:
                config.write(fileconfig)

        #EXPORT PARAMETER COMUNITY_LIST
        if self.parm_opt['all_comunities_bool'] == True:
            self.parm_opt['list_comunities'] = 'all'
        return self.parm_opt['list_comunities']
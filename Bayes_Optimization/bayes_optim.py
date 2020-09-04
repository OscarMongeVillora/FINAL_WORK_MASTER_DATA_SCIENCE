# BAYESIANOPTIMIZATION WITH CV FOR TEMPORAL SERIES-- TIME SPLITTING - GENERALIZATION
import pandas as pd
from GPyOpt.methods import BayesianOptimization
from Decompose.decompose_data import Decompose
from Models_ML.models_and_domains import ModelRegressor

#This main that connects the different modules to run the Bayes optimization tuning
# Defines the Decompose object, ModelRegressor object, Objective_Func object and BayesOptimization Object (of GPyOpt)
# It defines the Bayesian Optimization, fit the data to the optimal model and get the optimal prediction

from Bayes_Optimization.Func import Objective_Func
class Bayesian_Opt(object):
    def __init__(self, x_train_features, data_train, n_sample_subset, timesplitmethod,  model_choose, loss_function,
                 log_transform, fit_trend_method, parameters_trend, mode_dataset, percentage_validation,overlap,
                 bayes_tuning_on = True, predict_intervals = False):
        self.x_train_features = x_train_features
        self.data_train = data_train
        self.opt_train_size = len(self.x_train_features[:,0]) #2D columnar x_train_features
        self.n_sample_subset = n_sample_subset
        self.timesplitmethod = timesplitmethod
        self.model_choose = model_choose
        self.predict_intervals = predict_intervals
        self.loss_function = loss_function
        self.data_train_log_trend = data_train
        self.model_cv, self.domain, self.func, self.opt_search, self.predict_train, self.model_final = [None] * 6
        self.fitted = 0
        self.pred_mod_intervals = 0
        self.log_transform = log_transform
        self.fit_trend_method = fit_trend_method
        self.parameters_trend = parameters_trend
        self.mode_dataset = mode_dataset
        self.decompose_model = None
        self.percentage_validation = percentage_validation
        self.overlap = overlap
        self.bayes_tuning_on = bayes_tuning_on

    # CREATE THE DECOMPOSE OBJECT: return the times series without trend
    def descompose_train_data(self):
        self.decompose_model =Decompose(self.x_train_features, self.data_train, self.log_transform,
                                        self.fit_trend_method, self.parameters_trend, self.mode_dataset)
        self.data_train_log_trend = self.decompose_model.descompose_train_data()

    # DEFINES THE REGRESSOR MODEL CHOSEN AND ITS DOMAIN
    def choose_model(self):
        self.model_cv = ModelRegressor(self.model_choose, self.predict_intervals)
        self.domain = self.model_cv.domains_models()

    # DEFINES THE OBJECTIVE FUNCTION: it will be the input of the BayesianOptimization module (GpyOpt)
    def func_to_optimize(self):
        #Declaro el objeto Objective_Func importado
        self.func = Objective_Func(self.x_train_features, self.data_train, self.data_train_log_trend,
                                   self.n_sample_subset, self.timesplitmethod, self.opt_train_size,
                                   self.model_cv, self.loss_function, self.decompose_model,
                                   self.percentage_validation, self.overlap)

    # DEFINES THE BAYESIAN OPTIMIZATION OBJECT FOR THE EXTERNAL LIBRARY GPYOPT
    def Bayesdefinition(self):
        self.opt_search = BayesianOptimization(self.func.objective_function,
                                          domain=self.domain,
                                          model_type='GP',
                                          acquisition_type='EI',
                                          num_cores=-1,
                                          verbosity=False)

    # RUN THE BAYES OPTIMIZATION: call the four previous objects and after run the run_optimization method
    # Return the optimal parameters for the model chosen and the evaluations of the Bayes Optimization method.
    def run(self, max_iter=100):
        self.descompose_train_data()
        self.choose_model()
        self.func_to_optimize()

        if self.bayes_tuning_on == True:
            self.Bayesdefinition()
            self.opt_search.run_optimization(max_iter)
            print('The run has finished correctly')
            #plot = self.opt_search.plot_convergence()
            self.opt_param = self.opt_search.x_opt  #Get the optimal parameters
            dict_param = {sentence['name']: self.opt_param[i] for i, sentence in enumerate(self.domain)}
            evaluations = self.opt_search.get_evaluations()[0]  #Get the Bayes Optimization evaluations
            rmse_evaluations = self.opt_search.get_evaluations()[1]
            eval_columns = pd.DataFrame({sentence['name']: evaluations[:,i]  for i, sentence in enumerate(self.domain)})
            eval_columns['rmse'] = rmse_evaluations

        elif self.bayes_tuning_on == False:
            print('Bayesian Optimization is not activated')
            self.opt_param = [x['domain'][0] for x in self.domain ]
            dict_param = {sentence['name']: self.opt_param[i] for i, sentence in enumerate(self.domain)}
            eval_columns = []

        return [dict_param, eval_columns]

    #INITIALIZE THE MODEL CHOSEN WITH THE OPTIMAL PARAMETERS. It returns the final model object.
    def best_model(self):
        list_hyperparam_final = [hyperparam for hyperparam in self.opt_param]

        if self.predict_intervals == False:
            self.model_final = self.model_cv.regressor(list_hyperparam_final)

        # For this option we initialize 3 different models with the hyperparameters for three diferent quantiles.
        if self.predict_intervals == True:
            model_final_mid = self.model_cv.regressor(list_hyperparam_final)
            model_final_down = self.model_cv.regressor(list_hyperparam_final, pred_lower_upper = -1)
            model_final_up = self.model_cv.regressor(list_hyperparam_final, pred_lower_upper = 1)

            self.model_final = [model_final_mid, model_final_down, model_final_up]

        return self.model_final

    #  FIT THE DATA TO THE BEST MODEL
    def fit(self):
        if self.predict_intervals == False:
            self.best_model().fit(self.x_train_features, self.data_train_log_trend)

        # For this option we fit the 3 different models for the three diferent quantiles
        if self.predict_intervals == True:
            [model.fit(self.x_train_features, self.data_train_log_trend) for model in self.best_model()]

        self.fitted = 1
        print('The training fit is done')

    #MAKE THE PREDICTIONS ON THE OPTIMAL MODEL: returns the final prediction.
    # Enter x_feature as A 2D-columnar array where x_feature[:,0] is the series of days.
    def predict(self, x_features):
        if self.fitted == 1:
            final_prediction = None

            if self.predict_intervals == False:
                prediction_without_trend = self.model_final.predict(x_features).reshape(-1,1)
                final_prediction = self.decompose_model.predict_compose(x_features, prediction_without_trend)

            if self.predict_intervals == True:
                prediction_without_trend = [model.predict(x_features).reshape(-1, 1) for model in self.model_final]
                final_prediction_mid = self.decompose_model.predict_compose(x_features, prediction_without_trend[0])
                final_prediction_down = self.decompose_model.predict_compose(x_features, prediction_without_trend[1])
                final_prediction_up = self.decompose_model.predict_compose(x_features, prediction_without_trend[2])
                final_prediction = [final_prediction_mid, final_prediction_down, final_prediction_up]

            return final_prediction
        else:
            print('You need to fit the model to your data')











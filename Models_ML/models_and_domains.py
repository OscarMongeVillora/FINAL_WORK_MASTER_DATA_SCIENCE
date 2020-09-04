from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from Models_ML.adapted_SARIMA import SARIMA_model
from Models_ML.adapted_NN import NN_model

# This module creates the class ModelRegressor, which function regressor() defines the different ML methods
#to fit and make predictions

class ModelRegressor(object):

    def  __init__(self, model_choose, predict_intervals = False):
        self.model_choose = model_choose
        self.predict_intervals = predict_intervals

    #Defines the ML models objects. Receive as an input the list of hyperparmaters tried.
    def regressor(self, list_hyperparam_model, pred_lower_upper=0):

        model = None
        if self.predict_intervals == False:

            if self.model_choose == 'xgb':
                n_estimators, max_depth, learning_rate = list_hyperparam_model
                model = XGBRegressor(n_estimators = int(round(n_estimators)), max_depth = int(round(max_depth))
                                     , learning_rate = learning_rate)
                #model = GradientBoostingRegressor(loss='quantile', alpha=0.5, n_estimators=int(round(n_estimators)),
                                                  #max_depth=int(round(max_depth)), learning_rate=learning_rate)
                # model = XGBRegressor(n_estimators=30, max_depth=10,
                # learning_rate=0.14119048, reg_alpha=x[j, 0], reg_lambda=x[j, 1]

            elif self.model_choose == 'rf':

                n_estimators, max_depth= list_hyperparam_model
                model = RandomForestRegressor(n_estimators = int(round(n_estimators)),
                                              max_depth = int(round(max_depth)))

            #Don't recommended to use it.
            elif self.model_choose == 'svr':

                gamma, C, epsilon = list_hyperparam_model
                #model = Pipeline([('Scaler', StandardScaler()),
                                  #('SVR', SVR(kernel='rbf', gamma = gamma, C = C, epsilon = epsilon))])
                model = SVR(kernel='rbf', gamma = gamma, C = C, epsilon = epsilon)

            elif self.model_choose == 'SARIMA':
                param_estacionality = 7
                model = SARIMA_model(list_hyperparam_model, param_estacionality)

            elif self.model_choose == 'NN':
                param_epoch = 250
                train_window = 28
                model = NN_model(list_hyperparam_model, param_epoch, train_window)

        ############################################################################
        if self.predict_intervals == True:

            if self.model_choose == 'xgb':
                # Definition of the quantiles
                alpha = 0.1
                lower_alpha, upper_alpha = alpha, 1 - alpha
                n_estimators, max_depth, learning_rate = list_hyperparam_model

                #Definition of the mid model that uses the default loss: 'ls', which we apply Bayes Optimization
                if pred_lower_upper == 0:
                    model = GradientBoostingRegressor(loss = 'quantile', alpha = 0.5,
                                                      n_estimators = int(round(n_estimators)),
                                                      max_depth = int(round(max_depth)),
                                                      learning_rate = learning_rate)
                else:
                    if pred_lower_upper == -1:
                        alpha = lower_alpha
                    elif pred_lower_upper == 1:
                        alpha = upper_alpha
                    model = GradientBoostingRegressor(loss = 'quantile', alpha = alpha,
                                                      n_estimators=int(round(n_estimators)),
                                                      max_depth=int(round(max_depth)), learning_rate=learning_rate)
        return model

    #HYPERPARAMETER DOMAINS: Defined for each ML model and returned. Input of the BayesianOptimization method
    # to search the optimal ones.
    def domains_models(self):
        if self.model_choose == 'xgb':
            self.domain = [{'name': 'n_estimators', 'type': 'continuous', 'domain': (1, 300)},
                      {'name': 'max_depth', 'type': 'continuous', 'domain': (3, 40)},
                           {'name': 'learning_rate', 'type': 'continuous', 'domain':(0.01, 1.5)}]
            # xgboost with L1 L2
            #self.domain = [{'name': 'reg_alpha', 'type': 'continuous', 'domain': (0, 10)},
                      #{'name': 'reg_lambda', 'type': 'continuous', 'domain': (0, 10)}]
        if self.model_choose == 'rf':
            # Random Forest
            self.domain = [{'name': 'n_estimators', 'type': 'continuous', 'domain': (1, 300)},
                      {'name': 'max_depth', 'type': 'continuous', 'domain': (4, 40)}]

        if self.model_choose == 'svr':
            # SVR
            self.domain = [{'name': 'gamma', 'type': 'continuous', 'domain': (0.001, 100)},
                      {'name': 'C', 'type': 'continuous', 'domain': (0.001, 100)},
                      {'name': 'epsilon', 'type': 'continuous', 'domain': (0.001, 100)}]

        if self.model_choose == 'SARIMA':

            self.domain = [{'name': 'p', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
                      {'name': 'd', 'type': 'discrete', 'domain': (0, 1)},
                      {'name': 'q', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
                      {'name': 'P', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)},
                      {'name': 'D', 'type': 'discrete', 'domain': (0, 1)},
                      {'name': 'Q', 'type': 'discrete', 'domain': (0, 1, 2, 3, 4)}]

        if self.model_choose == 'NN':
            #learning rate
            self.domain = [{'name': 'lr', 'type': 'discrete', 'domain': (0.04904, 0.05)}]

            #self.domain = [{'name': 'train_window', 'type': 'discrete', 'domain': (28, 28)},
                           #{'name': 'lr', 'type': 'continuous', 'domain': (0.04, 0.06)}]

                      #{'name': 'lr', 'type': 'continuous', 'domain': (0.0001, 0.01)}]

        #self.domain = self.domain + [{'name': 'degree', 'type': 'continuous', 'domain': (1, 6)}]

        return self.domain
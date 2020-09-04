import numpy as np
from sklearn.metrics import mean_squared_error as mse
from Decompose.decompose_data import Decompose
from Utils.timesplit import times_split

#This script defines the Objective Function that is minimized by Bayes Optimization method.

class Objective_Func(object):
    def __init__(self, x_train_features, data_train, data_train_log_trend, n_sample_subset,
                 timesplitmethod, opt_train_size, model_cv, loss_function, decompose_model,
                 percentage_validation, overlap):

        self.x_train_features = x_train_features
        self.data_train = data_train
        self.data_train_log_trend = data_train_log_trend
        self.n_sample_subset = n_sample_subset
        self.timesplitmethod = timesplitmethod
        self.opt_train_size = opt_train_size
        self.model_cv = model_cv
        self.loss_function = loss_function
        self.decompose_model = decompose_model
        self.n_subsets = int(round(self.opt_train_size / self.n_sample_subset))
        self.percentage_validation = percentage_validation
        self.overlap = overlap

    # DEFINE OBJECTIVE FUNCTION
    # This is the function that will be the input of the external module GPyOpt.BayesOptimization at bayes_optim.py
    def objective_function(self, x):
        x = np.atleast_2d(x)
        fs = np.zeros((x.shape[0], 1))

        for j in range(x.shape[0]):
            fs[j] = 0

            # TIMES SERIES SPLIT (METHOD): we obtain the sample indexs for the subtrain and validation samples.
            for i in range(0, self.n_subsets):
                idx_subset_train, idx_subset_valid = times_split(self.opt_train_size, self.n_sample_subset,
                                                                 self.percentage_validation,self.overlap,
                                                                 self.timesplitmethod, iteration=i)

                # CHOOSE THE SAMPLES OF THE TRAINING DATA USING THE SUBTRAIN AND VALIDATION INDEXES
                x_train_subset = self.x_train_features[idx_subset_train, :]
                data_train_subset = self.data_train_log_trend.iloc[idx_subset_train]
                x_validation_subset = self.x_train_features[idx_subset_valid, :]
                data_validation_subset = self.data_train_log_trend.iloc[idx_subset_valid]

                # CHOOSE MODEL, FIT AND PREDICT : we input the hyperparameters tried in this iteration by the Bayes.Opt.
                # In every i,j iteration I define diferent regressor models(different data and hyperparameters as input)
                # For obtaining the recovered predictions in 'pred_subset_validation' I sum the trend and exp(series)
                list_hyperparam_model = [hyperparam for hyperparam in x[j, :]]
                model = self.model_cv.regressor(list_hyperparam_model)
                model.fit(x_train_subset, np.array(data_train_subset).reshape(-1, 1))
                pred_subset_validation = self.decompose_model.predict_compose(x_validation_subset,
                                                                              model.predict(x_validation_subset))

                # LOSS FUNCTION (we compare the recovered prediction with the real one for this validation subset)
                data_validation_subset_real = self.data_train.iloc[idx_subset_valid]
                rmse_error = np.sqrt(mse(data_validation_subset_real.values.squeeze(),pred_subset_validation.squeeze()))

                #SUM OF ALL THE ERRORS FOR ALL THE SUBSETS OF VALIDATION (After that, they are averaged)
                if self.loss_function == 0:
                    fs[j] += rmse_error
                if self.loss_function == 1:
                    R_error = 1 - (((rmse_error ** 2) * self.n_sample_subset) /
                                   sum(((data_validation_subset_real - np.mean(
                                       data_validation_subset_real)).values) ** 2))
                    fs[j] += -R_error

            fs[j] *= 1 / (i + 1)

        return fs

        # We have to return a 2D columnar array, being every row the evaluation of the loss function for one
        # array of hyperparameters tested.
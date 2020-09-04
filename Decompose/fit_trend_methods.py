import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from Datasets.datasets_prepare import diff_cum_conv
from Decompose.sir_equations import Sir_object
from matplotlib import pyplot as plt
import numpy as np

#This module defines the models for fitting the trend of the time series. It defines a generalized fit and predict.

#There are methods that will need to convert the dataset into differentiate or into cumulative to execute the fitting
# and after do the inverse transformation to compose the prediction.

class trend_method(object):
    def __init__(self, x_train_features, data_train, method_name, parameters_trend, mode_dataset):
        self.x_train_features = x_train_features
        self.data_train = data_train
        self.method = method_name
        self.parameters_trend = parameters_trend
        self.mode_dataset = mode_dataset
        self.trend = None
        self.poly_features = None
        self.line_reg = None
        self.exp_model = None
        self.logistic_model = None
        self.gompertz_model = None
        self.parameters_optim = None

    #FIT THE CHOOSEN MODEL TO THE DATA
    def fit(self):
        #############################
        if self.method == 'polynomic':
            poly_order = self.parameters_trend
            self.poly_features = PolynomialFeatures(poly_order)  # Adds a new feature of 'poly_order' degree to x_train
            x_poly = self.poly_features.fit_transform(self.x_train_features[:, 0].reshape(-1, 1))
            self.line_reg = LinearRegression()
            self.line_reg.fit(x_poly, self.data_train)
        #############################
        #Don't recommended to use this method. It doesn't work well with these time series.
        if self.method == 'exponential':
            if self.mode_dataset == 'diff':  # Convert the time series into cumulative time series
                self.data_train = diff_cum_conv(type='cum', data=self.data_train, axis=0)

            #Load the parameters x0, valued on the parameters_adjusted_dataset.py
            x0 = self.parameters_trend #Shifted days of the beginning.

            #It must take the independent parameter as the first parameter -- N0 * (1 + p) ** (x - x0)
            self.exp_model = lambda x, p:  self.data_train.iloc[0] * (1 + p) ** ((x - x0) - self.x_train_features[0, 0])
            def exp_model(x, p):
                return self.data_train.iloc[0] * (1 + p) ** ((x - x0) - self.x_train_features[0, 0])

            #Fit the curve  (p0 is the initial guess for the parameter)
            fit_model = curve_fit(exp_model, xdata = self.x_train_features[:,0].squeeze(),
                                  ydata = self.data_train.squeeze() , p0=[0.23], maxfev=10000)

            self.parameters_optim = fit_model[0]  # p are the optimal parameters
            sigma_p = np.sqrt(np.diag(fit_model[1]))  #It is the standard deviation for each parameter
        ##############################
        if self.method == 'logistic':
            if self.mode_dataset == 'diff': # Convert the time series into cumulative time series
                self.data_train = diff_cum_conv(type='cum', data=self.data_train, axis=0)

            # Load the parameters x0, valued on the parameters_adjusted_dataset.py
            x0 = self.parameters_trend
            self.logistic_model = lambda x, a, b, c:  c / (1 + np.exp(-((x - x0) - b) / a))

            def logistic_model(x,a,b,c):
                return c / (1 + np.exp(-((x - x0) - b) / a))

            # Fit the curve
            fit_model = curve_fit(logistic_model, xdata = self.x_train_features[:,0].squeeze(),
                                  ydata = self.data_train.values.squeeze(), method = 'lm', maxfev=1000000)

            self.parameters_optim = fit_model[0]
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(fit_model[1]))
        ###############################
        if self.method == 'gompertz':
            if self.mode_dataset == 'diff': # Convert the time series into cumulative time series
                self.data_train = diff_cum_conv(type='cum', data=self.data_train, axis=0)

            # Load the parameters x0, valued on the parameters_adjusted_dataset.py
            x0 = self.parameters_trend #Initial day
            self.gompertz_model = lambda x, a, b, c:  c * np.exp(-b * np.exp(-(x - x0) / a))

            def gompertz_model(x,a,b,c):
                return c * np.exp(-b * np.exp(-(x - x0) / a))

            # Fit the curve
            fit_model = curve_fit(gompertz_model, xdata = self.x_train_features[:,0].squeeze(),
                                  ydata = self.data_train.squeeze(), method = 'lm', maxfev=1000000)

            self.parameters_optim = fit_model[0]
            print(self.parameters_optim)
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(fit_model[1]))
        ###################################
        if self.method == 'sir_model':

            if self.mode_dataset == 'diff': # Convert the time series into cumulative time series
                self.data_train = diff_cum_conv(type='cum', data=self.data_train, axis=0)

            mode_R_list = ['ct', 'mitigation']

            mode_R = mode_R_list[1]
            sir_object = Sir_object(self.data_train)

            #Define the R0 mitigating function
            def R0_mitigating(t, r0=3, mu=1, r_bar=1.6):
                R0 = r0 * np.exp(- mu * (t)) + (1 - np.exp(- mu * (t))) * r_bar
                return R0

            #Define the SIR MODEL,takes the R values (input) and solve the differential equations at function solve_path
            def sir_model(x, r0, mu, r_bar):
                if mode_R == 'mitigation':
                    R0 = lambda t: R0_mitigating(t, r0, mu, r_bar)

                elif mode_R == 'ct':
                    R0 = r0
                t_vec = x
                c_path = sir_object.solve_path(R0, t_vec)

                return c_path #Return the cumulative cases

            #Fit the curve
            self.sir_model = lambda x, r0, mu, r_bar: sir_model(x, r0, mu, r_bar)
            fit_model = curve_fit(sir_model, xdata=self.x_train_features[:,0].squeeze(),
                                  ydata=self.data_train.values.squeeze(), method = 'lm', maxfev=1000000)

            #Obtain the best values for parameters of R curve
            self.parameters_optim = fit_model[0]
            r0, mu, r_bar= self.parameters_optim

            #I generate the optimal trend for all the time series
            t_full = np.linspace(0, 399, 400)
            self.full_solution = self.sir_model(t_full, r0, mu, r_bar)

            if mode_R == 'mitigation':
                delay = 17
                R0_plot = lambda t: R0_mitigating(t + delay, r0, mu, r_bar)
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(t_full, R0_plot(t_full), label='R')
                ax.legend()
            if mode_R == 'ct':
                R0 = r0
                fig, ax = plt.subplots(figsize=(15, 5))
                ax.plot(t_full, [r0] * len(t_full), label='R')
                ax.legend()


        if self.method == 'None':
            self.parameters_optim = 0


    #PREDICT THE TREND ON THE TEST PART FOR THE MODEL CHOSEN
    def predict(self, x_features):
        if self.method == 'polynomic':
            x_poly = self.poly_features.fit_transform(x_features[:, 0].reshape(-1, 1))
            self.trend = self.line_reg.predict(x_poly)

        if self.method == 'exponential':
            self.trend = self.exp_model(x_features[:, 0].squeeze(), self.parameters_optim)

            # We convert it again into the trend in differential version
            if self.mode_dataset == 'diff':
                self.trend = diff_cum_conv(type='diff', data=self.trend, axis=0)

        if self.method == 'logistic':
            a, b, c = self.parameters_optim   #Use the optimal parameters found in the fitting
            self.trend = self.logistic_model(x_features[:, 0].squeeze(), a, b, c)

            # We convert it again into the trend in differential version
            if self.mode_dataset == 'diff':
                self.trend = diff_cum_conv(type='diff', data=self.trend, axis=0)

        if self.method == 'gompertz':
            a, b, c = self.parameters_optim  #Use the optimal parameters found in the fitting
            self.trend = self.gompertz_model(x_features[:, 0].squeeze(), a, b, c)

            # We convert it again into the trend in differential version
            if self.mode_dataset == 'diff':
                self.trend = diff_cum_conv(type='diff', data=self.trend, axis=0)

        if self.method == 'sir_model':
            delay = 0
            t_pred = x_features[:,0].squeeze()
            #print(t_pred)
            #Select the trend prediction of the days of interest
            self.trend = self.full_solution[t_pred[0] + delay: t_pred[-1] + 1 + delay]

            if self.mode_dataset == 'diff':
                self.trend = diff_cum_conv(type='diff', data=self.trend, axis=0)

        if self.method == 'None':
            self.trend = np.zeros(len(x_features))



        return self.trend
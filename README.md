# FINAL WORK MASTER DATA_SCIENCE 
This is the final work for the Master in Data Science of KSCHOOL.

Delivered day: 04 September 2020.

READ THE DOCUMENT "MEMORIA_TFM_OSCAR_MONGE_DATA_SCIENCE_FINAL_VERSION.pdf"


**Abstract**
 
The goal of this work is to compare the performance of different machine learning methods implemented in a proper way for predicting COVID-19 time series (Cases and Deaths). 

In order to make predictions for Covid-19 time series, a program has developed integrating different modules that automatize the different steps of the process: choosing dataset of Covid-19, cleaning and preparing the data; eliminating the heteroscedasticity and trend; fitting and predicting using a machine learning method; tuning the result with Bayesian Optimization based on Time Splitting Cross Validation and obtaining different plots and metrics errors as a result. 

The main program uses different approaches taken of the literature as Logistic and Gompertz fitting, SIR Model, etc. Furthermore, the program can be adapted for any temporal series and it allows to configure multiple parameters for the prediction of time series. 

Using that tool, we execute some simulations for different approaches to compare the performance of the ML Models and visualize some predictions.  
 
 

**Structure of the project**

The project is divided in 6 modules: Datasets, Bayes Optimization, Decompose, Models ML, Utils and Plots and Error. The input of the system are the datasets of time series of COVID-19 (Datasets module) and the config_generator.py file that allows to choose different option configurations. The output are the day predictions in the COVID-19 temporal series returned through plots and errors calculated using the Plots and Errors module. 

The project runs since __main__.py calling the module main_project.py, that contains the main instructions of the project. Basically, it imports the configuration parameters, prepares the data of the time series and run the Bayes Optimization module to obtain an accurate prediction of the series. After that, it makes the instructions to plot and evaluate it in terms of error. 
 
 
 
**LIBRARY REQUIRED FOR EXECUTING THE PROGRAM**

GPy, GPyOpt, sklearn, numpy, matplotlib, pandas, statsmodels, configparser 

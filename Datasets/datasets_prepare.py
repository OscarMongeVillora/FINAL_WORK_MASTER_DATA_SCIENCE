import numpy as np
import pandas as pd

#This module is responsible of reading some COVID-19 datasets that has been download and saved on Datasets folder.
#It obtains and prepares the data as a data frame in a predetermined and common form: using the rows for the different
#Communities or Regions and the columns for the cases or death temporal series
#The mode indicate if we want the dataframe exported in cumulative mode or differentiate mode.

#RETURNS THE DATAFRAME PREPARED OF THE DATASET CHOSEN
def import_datasets(name: str, mode = 'diff'):
    data_clean = None

    if name == 'deaths_old':
        #DATASET 1 Deaths for each community of Spain (Updated in May)
        data_deaths = pd.read_csv('Datasets/ccaa_covid19_fallecidos.csv', index_col='CCAA')
        data_deaths.drop(columns='cod_ine', inplace=True)
        #   decide the data period: 12/03/2020 - 20/05/2020 --  70 samples
        data_deaths = data_deaths.iloc[:, 7:-4]

        #Decide the mode of the dataset
        if mode == 'diff':
            data_clean = diff_cum_conv(type= 'diff', data = data_deaths, axis = 1) #Convert the dataset to incremental
        elif mode == 'cum':
            data_clean = data_deaths


    elif name == 'deaths_update':
        #DATASET 2 Deaths for each community of Spain (Updated in July)
        data_deaths_act = pd.read_csv('Datasets/ccaa_covid19_fallecidos_por_fecha_defuncion_nueva_serie.csv',
                                      index_col='CCAA')
        data_deaths_act.drop(columns='cod_ine', inplace=True)
        #   decide the data-period: 07/03/2020 - 23/07/2020 --  139 Samples (After doing the slicing)
        data_deaths_act = data_deaths_act.iloc[:, 23:123]
        if mode == 'diff':
            data_clean = data_deaths_act
        elif mode == 'cum':
            data_clean = diff_cum_conv(type= 'cum', data = data_deaths_act, axis = 1)

    elif name == 'casos_update':
        #DATASET 3 Cases for each community of Spain (Updated in July)
        data_casos_act = pd.read_csv('Datasets/ccaa_covid19_datos_isciii_nueva_serie.csv', parse_dates=['fecha'])
        data_casos_act.drop(columns='cod_ine', inplace=True)
        data_casos_act = data_casos_act.sort_values(['ccaa', 'fecha'])[['fecha', 'ccaa', 'num_casos']]
        #We want always as output the comunities in the different rows
        data_casos_act = data_casos_act.pivot(index='ccaa', columns='fecha', values = 'num_casos')
        #   decide the data-period: 01/31/2020 - 19/07/2020 --  171 Samples
        if mode == 'diff':
            data_clean = data_casos_act
        elif mode == 'cum':
            data_clean = diff_cum_conv(type= 'cum', data = data_casos_act, axis = 1)

    elif name == 'national':
        #DATASET 4 Spanish Cases. Time Series not corrected (deaths or cases)
        data_nac = pd.read_csv('../../../datasets_posibles_uso/nacional_covid19.csv', parse_dates=['fecha'],
                               index_col='fecha')
        data_nac.fillna(0, inplace=True)
        data_nac['casos_total'] = data_nac['casos_pcr'] + data_nac['casos_test_ac']
        if mode == 'diff':
            data_nac = diff_cum_conv(type= 'diff', data = data_nac, axis = 0)
        elif mode == 'cum':
            data_nac = data_nac
        data_nac = data_nac[['casos_total']]  #Podría haber elegido 'altas' o 'fallecimientos'
        #   decide the data-period: 22/02/2020 - 18/05/2020 --  87 Samples
        data_nac = data_nac.loc[:'2020-05-18']
        data_nac = data_nac.T
        data_clean = data_nac

    elif name == 'worldwide':
        #Cases or Deaths for each country of the world
        variable = 'initial'
        SELECT_REGION = 'world'
        data4 = pd.read_csv('Datasets/WHO-COVID-19-global-data.csv', parse_dates=['Date_reported'])
        if mode == 'diff':
            variable = ' New_deaths'  #To select the cases, replace the variable by ' New_cases'
        if mode == 'cum':
            variable = ' Cumulative_deaths' #To select the cases, replace the variable by ' Cumulative_cases'

        # Sum the time series for all the countries of the world, matching the dates.
        dict = {}
        for i, state in enumerate(data4[[' Country_code']].values.squeeze()):
            dict[state] = 1
        all_countries = {}

        if SELECT_REGION in dict.keys():  # We select only one country instead of all the world
            dict = {}
            dict[SELECT_REGION] = 0

        for i, country in enumerate(dict.keys()):

                data5 = (data4[[' Country_code']] == country)
                data6 = data5[data5].dropna()
                data_com = data4.iloc[data6.index]
                dat_com = data_com.set_index('Date_reported')
                dat_com = dat_com.ewm(7.).mean()

                for data in dat_com.index:

                    if data in all_countries.keys():
                        all_countries[data] += dat_com[[variable]].loc[data]
                    else:
                        all_countries[data] = dat_com[[variable]].loc[data]

        world_series = pd.DataFrame(all_countries)
        world_series = world_series.T
        world_series.reset_index(inplace=True)
        world_series.sort_values('index', inplace=True)
        world_series.set_index('index', inplace=True)
        world_series = world_series.iloc[0:-1]
        data_clean = world_series.T



    return data_clean


#CONVERTER CUMULATIVE-DIFFERENTIAL MODE
def diff_cum_conv(type: str, data, axis: int):
    data = pd.DataFrame(data)
    #axis = 1 we differentiate by elements of the row, axis = 0 we differentiate by elements of the column
    if type == 'diff':
        data_new = data.diff(axis = axis)
        data_new.fillna(0, inplace=True)  #Así no cambio el length
        #data_new.dropna(axis, inplace=True)

    if type == 'cum':
        data_new = data.cumsum(axis = axis)

    return data_new
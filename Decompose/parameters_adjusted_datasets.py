
#   We create the dictionary with the different comunities with its proper order of polinonomy for the trend

def load_parameters_adjusted(name_dataset, method_trend_name, data_clean):
    name = name_dataset
    method_trend = method_trend_name
    all_parameters_com = dict()   #It is a dictionary made to take different parameters (values) for different regions (keys)
    #If a put more than one parameter, I need to create the dit introducing a list as values, containing the parameters for the same comunity.
    if name == 'deaths_old':
        if method_trend == 'polynomic':
            #First and only parameter : poly_order
            list_comunities_o4 = ['Andalucía', 'C. Valenciana', 'Madrid', 'País Vasco',
                                  'Castilla La Mancha', 'Castilla y León', 'Cataluña', 'Extremadura']
            list_comunities_o3 = ['Aragón', 'Asturias', 'Baleares', 'Canarias', 'Cantabria',
                                  'Galicia', 'Melilla', 'Murcia', 'Navarra', 'La Rioja']


            poly_order_all_com = {x :4 for x in list_comunities_o4}
            poly_order_all_com2 = {x :3 for x in list_comunities_o3}
            poly_order_all_com.update(poly_order_all_com2)
            all_parameters_com = poly_order_all_com

        if method_trend == 'exponential':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index}
            all_parameters_com = x0_all_com

        if method_trend == 'logistic':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index}
            all_parameters_com = x0_all_com

        if method_trend == 'gompertz':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index}
            all_parameters_com = x0_all_com

        if method_trend == 'sir_model':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index}
            all_parameters_com = x0_all_com
        if method_trend == 'None':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index}
            all_parameters_com = x0_all_com
    elif name == 'deaths_update':
        if method_trend == 'polynomic':
            poly_order_all_com = {x: 2 for x in data_clean.index if x != 'España'}
            all_parameters_com = poly_order_all_com

        if method_trend == 'exponential':
            x0 = 23
            x0_all_com = {x: x0 for x in data_clean.index if x != 'España'}
            all_parameters_com = x0_all_com

        if method_trend == 'logistic':
            x0 = 23
            x0_all_com = {x: x0 for x in data_clean.index if x != 'España'}
            all_parameters_com = x0_all_com

        if method_trend == 'gompertz':
            x0 = 23
            x0_all_com = {x: x0 for x in data_clean.index if x != 'España'}
            all_parameters_com = x0_all_com

        if method_trend == 'sir_model':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index if x != 'España'}
            all_parameters_com = x0_all_com

        if method_trend == 'None':
            x0 = 7
            x0_all_com = {x: x0 for x in data_clean.index if x != 'España'}
            all_parameters_com = x0_all_com

    elif name == 'casos_update':
        #if method_trend == 'polynomic':
        poly_order_all_com = {x:5 for x in data_clean.index}
        all_parameters_com = poly_order_all_com

    elif name == 'national':
        if method_trend == 'polynomic':
            poly_order_all_com = {x:5 for x in data_clean.index}
            all_parameters_com = poly_order_all_com

    elif name == 'worldwide':
        if method_trend == 'polynomic':
            poly_order_all_com = {'world':7}
            all_parameters_com = poly_order_all_com

        if method_trend == 'gompertz':  #Estaba en 7
            x0 = 7
            x0_all_com = {'world': x0}
            all_parameters_com = x0_all_com

        if method_trend == 'logistic':
            x0 = 7
            x0_all_com = {'world': x0}
            all_parameters_com = x0_all_com

        if method_trend == 'sir_model':
            x0 = 7
            x0_all_com = {'world': x0}
            all_parameters_com = x0_all_com

        if method_trend == 'None':
            x0 = 7
            x0_all_com = {'world': x0}
            all_parameters_com = x0_all_com

    return all_parameters_com
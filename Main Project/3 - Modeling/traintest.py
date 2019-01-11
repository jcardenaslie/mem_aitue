import numpy as np
import pandas as pd

##############################################################################
# PREDICTORS
from predictors import predictors_set

from scipy.stats import randint

from sklearn.model_selection import train_test_split
##############################################################################

import warnings
warnings.filterwarnings('ignore')

from settings import base_settings

#####################################################################################
#SETTINGS
if __name__ == '__main__':

    print(base_settings['N_JOBS'], 
        base_settings['RANDOM_STATE'], 
        base_settings['DATASET_NRO'], 
        base_settings['EXPERIMENT_NRO'], 
        base_settings['PROYECT_SELECT'], 
        base_settings['USE_PREDICTORS'], 
        base_settings['NO_NULLS'], 
        base_settings['SESGO'], 
        base_settings['TARGET_PREDICT'])
    ############################################################################################################################
    # DATA

    #FOR CHECKING COLUMNS
    # personas10 = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(dataset) , index_col=[0], encoding = "ISO-8859-1")
    # personas8 = pd.read_csv('..\\..\\Datos\\experiments\\personas_cotizacion8.csv' , index_col=[0], encoding = "ISO-8859-1")

    personas = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(base_settings['DATASET_FILE_NAME']), index_col=[0], encoding = "ISO-8859-1")
    personas_info = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(base_settings['DATASET_FILE_NAME']), index_col=[0], encoding = "ISO-8859-1")

    print('Personas Original Shape', personas.shape)
    print('Personas Original Copy Shape',personas_info.shape)

    # ########################################################################################################
    # # Se pasan las variables categoricas que son objetos a variables categoricas
    personas['loc_comuna'] = personas['loc_comuna'].astype('category')
    personas['loc_provincia'] = personas['loc_provincia'].astype('category')
    personas['loc_region'] = personas['loc_region'].astype('category')
    personas['tipo_cliente'] = personas['tipo_cliente'].astype('category')
    personas['sexo'] = personas['sexo'].astype('category')
    personas['medio_inicial'] = personas['medio_inicial'].astype('category')
    personas['actividad'] = personas['actividad'].astype('category')

    personas_info['loc_comuna'] = personas_info['loc_comuna'].astype('category')
    personas_info['loc_provincia'] = personas_info['loc_provincia'].astype('category')
    personas_info['loc_region'] = personas_info['loc_region'].astype('category')
    personas_info['tipo_cliente'] = personas_info['tipo_cliente'].astype('category')
    personas_info['sexo'] = personas_info['sexo'].astype('category')
    personas_info['medio_inicial'] = personas_info['medio_inicial'].astype('category')
    personas_info['actividad'] = personas_info['actividad'].astype('category')


    # Personas que solo cotizaron en un proyecto especifico
    # Para E5D5, E6D5 #########################################
    if base_settings['PROYECT_SELECT'] != 'All Proyects':
        mask = (personas[base_settings['PROYECT_SELECT']] > 0)
        personas = personas[mask]
        
        mask = (personas_info[base_settings['PROYECT_SELECT']] > 0)
        personas_info = personas_info[mask]


    #################################################################################################################

    print('Personas Original Filter Shape',personas.shape)
    print('Personas Original Filter Copy Shape',personas_info.shape)

    predictors = predictors_set[base_settings['USE_PREDICTORS']]

    if base_settings['NO_NULLS']:
        personas.replace(['sin informacion'], np.nan, inplace=True)
        personas.dropna(inplace=True)


    # ##############################################################################################
    # CAMBIO VARIABLE TARGET
    # Compra como variable objetivo
    
    # Cambio de [True, False] a [1, -1] para que salgan bien los resultados de la conf matrix

    if base_settings['TARGET_PREDICT'] == 'negocio':
        p_target = pd.DataFrame(personas.negocio)
        p_target.negocio = [1 if x == True else -1 for x in p_target.negocio] # EDITABLE
        p_target['target'] = [1 if x == True else -1 for x in p_target.negocio] # EDITABLE
    elif base_settings['TARGET_PREDICT'] == 'compra':
        p_target = pd.DataFrame(personas.compra)
        p_target.compra = [1 if x == True else -1 for x in p_target.compra] # EDITABLE
        p_target['target'] = [1 if x == True else -1 for x in p_target.compra] # EDITABLE

    p_target.head()
    
    #################################################################################################################
    # SELECCIONANDO PREDICTORES
    personas = personas[predictors]

    ###############################################################################################################
    # Corroborando que los largos de personas y sus variables objetivos sean iguales
    print('Target Shape', p_target.shape)
    print('Personas Dummies Shape', personas.shape)

    ############################################################################################################
    # TRANSOFRM DATA SET
    personas = pd.get_dummies(personas)
    print('Personas Dummies Shape', personas.shape)
    # print('Personas Dummies Columns', '\n',personas.columns.tolist())

    RANDOM_STATE, N_JOBS = base_settings['RANDOM_STATE'], base_settings['N_JOBS']
    FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS = base_settings['FOLDER_RESULTS'], \
    base_settings['TARGET_PREDICT'], base_settings['USE_PREDICTORS']
    
    #################################################################################################################
    # TRAIN TEST VALIDATION
    X = personas 
    col_predictors = X.columns
    y = p_target.target

    #_grid para hacer cross_validation y _test_o para hacer validacion sobre datos no vistos
    X_grid, X_test_o, y_grid, y_test_o = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    print('X Grid Shape',X_grid.shape)
    print('X Test Shape',  X_test_o.shape)
    print('Y Grid Shape',y_grid.shape )
    print('Y Test Shape', y_test_o.shape)
    print(type(X_grid), type(y_grid))

    # SAVE TRAIN & TEST
    X_grid.to_csv('{}//x_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS))
    print('X Grid Saved')
    y_grid.to_csv('{}//y_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))
    print('y Grid Saved')
    X_test_o.to_csv('{}//x_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))
    print('X Test Saved')
    y_test_o.to_csv('{}//y_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))
    print('y Test Saved')

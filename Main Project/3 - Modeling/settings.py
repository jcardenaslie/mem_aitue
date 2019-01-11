import os

base_settings = {
'N_JOBS' : 6, # -1 ocupa todo los cores
'RANDOM_STATE' : 42, # para que sea reproducible
'DATASET_NRO' : 10,
'EXPERIMENT_NRO' : 1,
# 'All Proyects', 'Altos del Valle', 'Edificio Urban 1470','San Andres Del Valle', 'Edificio Mil610', 'Edificio Junge'
'PROYECT_SELECT' : 'san andres del valle', # 
'USE_PREDICTORS' : 'isno', 
'TARGET_PREDICT' : 'compra',
'NO_NULLS' : False, #cambiable
'SESGO' : False, #cambiable
'SCORER' : 'f1_score', #f1_score precision
'CLASSES': ['Negocio', 'No Negocio']
}


EXP_NAME = "{} {}".format(base_settings['PROYECT_SELECT'],
	base_settings['USE_PREDICTORS'])
dataset_file_name = 'personas_cotizacion{}.csv'.format(base_settings['DATASET_NRO'])

#FOLDER CREATION
base_folder = 'resultados'
if not os.path.exists(base_folder):
    os.mkdir(base_folder)
    print('folder {} created'.format(base_folder))
      
predictores = base_settings['USE_PREDICTORS']
target_variable = base_settings['TARGET_PREDICT']

if not os.path.exists('{}//{}_{}'.format(base_folder, predictores, target_variable)):
    os.mkdir('{}//{}_{}'.format(base_folder, predictores, target_variable))
    print('folder {}\\{} created'.format(base_folder, predictores, target_variable))

FOLDER_RESULTS = '{}\\{}_{}'.format(base_folder, predictores, target_variable)
# print('FOLDER:', FOLDER_RESULTS)

base_settings['EXP_NAME'] = EXP_NAME
base_settings['DS_NAME'] = EXP_NAME
base_settings['DATASET_FILE_NAME'] = dataset_file_name
base_settings['FOLDER_RESULTS'] = FOLDER_RESULTS

###############################################################

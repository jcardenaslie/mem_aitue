import os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import shap
import copy

##############################################################################
# PREDICTORS
from predictors import predictors_set


import utils
##############################################################################
from sklearn.model_selection import (train_test_split, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold, StratifiedShuffleSplit)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, precision_recall_fscore_support,
    make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import svm
from xgboost import XGBClassifier
import itertools
from scipy.stats import randint

def view(i, r):
    print(
        'cv: ', i, '\n', 'Test'
        ' acc %.4f' % r['ts_acc'],
        '+- %.4f'% r['ts_acc-std'],
        '| pre %.4f'% r['ts_pre'],
        '+- %.4f'% r['ts_pre-std'],
        '| rec %.4f'% r['ts_rec'],
        '+- %.4f'% r['ts_rec-std'],
        '| fs %.4f'% r['ts_fs'],
        '+- %.4f'% r['ts_fs-std'],
        '| auc %.4f'% r['ts_auc'],
        '+- %.4f'% r['ts_auc-std'],
        '\n','Train',
        'acc %.4f' % r['tr_acc'],
        '+- %.4f'% r['tr_acc-std'],
        '| pre %.4f'% r['tr_pre'],
        '+- %.4f'% r['tr_pre-std'],
        '| rec %.4f'% r['tr_rec'],
        '+- %.4f'% r['tr_rec-std'],
        '| fs %.4f'% r['tr_fs'],
        '+- %.4f'% r['tr_fs-std'],
        '| auc %.4f'% r['tr_auc'],
        '+- %.4f'% r['tr_auc-std'],
        '\n'
        )

#####################################################################################
#SETTINGS
if __name__ == '__main__':
    N_JOBS = 6 # -1 ocupa todo los cores
    RANDOM_STATE = 42 # para que sea reproducible
    DATASET_NRO = 10
    EXPERIMENT_NRO = 1

    # 'All Proyects', 'Altos del Valle', 'Edificio Urban 1470','San Andres Del Valle', 'Edificio Mil610', 'Edificio Junge'
    PROYECT_SELECT = 'san andres del valle' #cambiable

    # base, base_profesion, base_minusisprofesion, base_medini_isrec
    USE_PREDICTORS = 'isno' #cambiable
    # experimento 2 y 4 va con True
    NO_NULLS = False #cambiable
    SESGO = False #cambiable

    # nombre experimento
    EXP_NAME = "{} {}".format(PROYECT_SELECT, USE_PREDICTORS)
    # eleccion dataset
    dataset_file_name = 'personas_cotizacion{}.csv'.format(DATASET_NRO)

    TARGET_PREDICT = 'negocio'

    DS_NAME = EXP_NAME


    print(N_JOBS, RANDOM_STATE, DATASET_NRO, EXPERIMENT_NRO, 
        PROYECT_SELECT, USE_PREDICTORS, NO_NULLS, SESGO, TARGET_PREDICT)
    ###############################################################

    #FOLDER CREATION
    base_folder = 'resultados'
    if not os.path.exists(base_folder):
        os.mkdir(base_folder)
        print('folder {} created'.format(base_folder))
        
    predictores = USE_PREDICTORS
    if not os.path.exists('{}\\{}'.format(base_folder, predictores)):
        os.mkdir('{}\\{}'.format(base_folder, predictores))
        print('folder {}\\{} created'.format(base_folder, predictores))

    target_variable = TARGET_PREDICT
    if not os.path.exists('{}\\{}\\{}'.format(base_folder, predictores, target_variable)):
        os.mkdir('{}\\{}\\{}'.format(base_folder, predictores, target_variable))
        print('folder {}\\{}\\{} created'.format(base_folder, predictores, target_variable))


    FOLDER_RESULTS = '{}\\{}\\{}'.format(base_folder, predictores, target_variable)

    print('FOLDER:', FOLDER_RESULTS)
    ############################################################################################################################
    # DATA

    #FOR CHECKING COLUMNS
    # personas10 = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(dataset) , index_col=[0], encoding = "ISO-8859-1")
    # personas8 = pd.read_csv('..\\..\\Datos\\experiments\\personas_cotizacion8.csv' , index_col=[0], encoding = "ISO-8859-1")

    personas = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(dataset_file_name), index_col=[0], encoding = "ISO-8859-1")
    personas_info = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(dataset_file_name), index_col=[0], encoding = "ISO-8859-1")

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
    if PROYECT_SELECT != 'All Proyects':
        mask = (personas[PROYECT_SELECT] > 0)
        personas = personas[mask]
        
        mask = (personas_info[PROYECT_SELECT] > 0)
        personas_info = personas_info[mask]


    #################################################################################################################

    print('Personas Original Filter Shape',personas.shape)
    print('Personas Original Filter Copy Shape',personas_info.shape)

    predictors = predictors_set[USE_PREDICTORS]

    if NO_NULLS:
        personas.replace(['sin informacion'], np.nan, inplace=True)
        personas.dropna(inplace=True)


    # ##############################################################################################
    # CAMBIO VARIABLE TARGET
    # Compra como variable objetivo
    p_target = pd.DataFrame(personas.negocio) # Target EDITABLE
    p_target.head()
    # Cambio de [True, False] a [1, -1] para que salgan bien los resultados de la conf matrix

    if TARGET_PREDICT == 'negocio':
        p_target.negocio = [1 if x == True else -1 for x in p_target.negocio] # EDITABLE
        p_target['target'] = [1 if x == True else -1 for x in p_target.negocio] # EDITABLE
    elif TARGET_PREDICT == 'compra':
        p_target.compra = [1 if x == True else -1 for x in p_target.compra] # EDITABLE
        p_target['target'] = [1 if x == True else -1 for x in p_target.compra] # EDITABLE

    #################################################################################################################
    # SELECCIONANDO PREDICTORES
    personas = personas[predictors]

    ###############################################################################################################
    # Corroborando que los largos de personas y sus variables objetivos sean iguales
    print('Target Shape', p_target.shape)
    print('Personas Dummies Shape', personas.shape)
    # both = set(personas.index) & set(p_target.index)
    # print('Len', len(both))
    # print(personas.shape)


    ############################################################################################################
    # TRANSOFRM DATA SET
    personas = pd.get_dummies(personas)
    print('Personas Dummies Shape', personas.shape)
    print('Personas Dummies Columns', '\n',personas.columns.tolist())

    #################################################################################################################
    # SUPERVISED LEARNING

    X = personas 
    col_predictors = X.columns
    y = p_target.target

    #_grid para hacer cross_validation y _test_o para hacer validacion sobre datos no vistos
    X_grid, X_test_o, y_grid, y_test_o = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    # X_grid, X_test_o, y_grid, y_test_o, indices_train, indices_test = train_test_split(X, y \
    #                                                     , indices,test_size=0.2, random_state=RANDOM_STATE)
    print('X Grid Shape',X_grid.shape)
    print('X Test Shape',  X_test_o.shape)
    print('Y Grid Shape',y_grid.shape )
    print('Y Test Shape', y_test_o.shape)
    print(type(X_grid), type(y_grid))

    # SAVE TRAIN & TEST
    # X_test_o.to_excel('x_test_compra_{}.xlsx'.format(USE_PREDICTORS))
    # y_test_o.to_excel('y_test_compra_{}.xlsx'.format(USE_PREDICTORS))
    X_test_o.to_csv('{}//x_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS))
    y_test_o.to_csv('{}//y_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))
    X_test_o.to_csv('{}//x_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))
    y_test_o.to_csv('{}//y_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS))

    #################################################################################################################
    clfs = dict()
    clf_params = dict()
    clfs_results = dict()

    clfs['LR'] = LogisticRegression(random_state=RANDOM_STATE)
    clf_params['LR'] = {'C': np.logspace(-5, 8, 15), 'penalty':['l1', 'l2'], 'class_weight':[None, 'balanced']}
    clfs_results['LR'] = dict()

    # clfs['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    # clf_params['DT'] = {"max_depth": range(1,20),"max_features":range(1,30), 
    #                     "min_samples_leaf": range(1,10),"criterion": ["gini", "entropy"], 
    #                     'class_weight':[None, 'balanced']}
    # clfs_results['DT'] = dict()

    # clfs['RF'] = RandomForestClassifier(random_state=RANDOM_STATE)
    # clf_params['RF'] = {"max_depth": range(1,20),"max_features":range(1,30), "min_samples_leaf": range(1,10),
    #                     "criterion": ["gini", "entropy"], 'class_weight':[None, 'balanced']}
    # clfs_results['RF'] = dict()

    # clfs['SVM'] = svm.SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    # clf_params['SVM'] = {'C' :[0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'class_weight':[None, 'balanced']}
    # clfs_results['SVM'] = dict()

    
    
    # clfs['XGB'] = XGBClassifier(random_state=RANDOM_STATE)
    # clf_params['XGB'] = {'n_estimators' : [50, 100, 150, 200], 'max_depth' : [2, 4, 6, 8]}
    # clfs_results['XGB'] = dict()

    ######################################################################################################################
    # clfs['KNN'] = KNeighborsClassifier()
    # clfs['PCAKNN'] = Pipeline([('pca', PCA(random_state=RANDOM_STATE)), ('clf', KNeighborsClassifier())])
    # clf_params['KNN'] = {'n_neighbors': np.arange(1, 20)}
    # clf_params['PCAKNN'] = {'pca__n_components': [2, 3, 4, 5, 6, 7, 8, 9]}
    # clfs_results['PCAKNN'] = dict()
    # clfs_results['KNN'] = dict()
    #########################################################################################################################

    scoring = {'accuracy' : make_scorer(accuracy_score), 
               # 'precision' : make_scorer(precision_score),
               'recall' : make_scorer(recall_score), 
               # 'f1_score' : make_scorer(f1_score),
              'auc':make_scorer(roc_auc_score)}

    for clf in clfs.keys():
        scorer = 'recall'
        print("Classifier:",clf, ' Scorer:', scorer)
        for i in range(3,4):

            kf = StratifiedKFold(n_splits=i, shuffle=True, random_state=RANDOM_STATE)

            if clf == 'RF' or clf == 'DT':
                grid = RandomizedSearchCV(clfs[clf], clf_params[clf], \
                                          cv=kf, n_jobs=N_JOBS, n_iter=300, \
                                          random_state=RANDOM_STATE, scoring=scoring, refit=scorer \
                                          ,return_train_score=True)
            else:
                grid = GridSearchCV(clfs[clf], clf_params[clf], cv=kf, n_jobs=N_JOBS\
                                    , scoring=scoring, refit=scorer, return_train_score=True)
            
            grid.fit(X_grid, y_grid)

            clfs_results[clf][i] = utils.grid_results(grid, scorer)
            r = clfs_results[clf][i]
            view(i, r)
        
        print('\n')

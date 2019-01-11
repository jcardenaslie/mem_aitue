import utils

import os
import numpy as np
import pandas as pd

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

import utils
from settings import base_settings
import warnings
from sklearn.externals import joblib
warnings.filterwarnings('ignore')

#View model results
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

if __name__ == '__main__':
#################################################################################################################
    RANDOM_STATE, N_JOBS = base_settings['RANDOM_STATE'], base_settings['N_JOBS']
    FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS = base_settings['FOLDER_RESULTS'], \
    base_settings['TARGET_PREDICT'], base_settings['USE_PREDICTORS']

    ############################################################################################################
    
    X_grid = pd.read_csv('{}//x_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS), index_col=[0])
    y_grid = pd.read_csv('{}//y_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0], header=None)
    X_test_o = pd.read_csv('{}//x_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0])
    y_test_o = pd.read_csv('{}//y_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0], header=None)
    # y_grid, y_test_o = pd.Series(y_grid), pd.Series(y_test_o)

    print('X Grid Shape',X_grid.shape)
    print('X Test Shape',  X_test_o.shape)
    print('Y Grid Shape',y_grid.shape )
    print('Y Test Shape', y_test_o.shape)
    print(type(X_grid), type(y_grid))

    #################################################################################################################
    # SUPERVISED LEARNING
    clfs = dict()
    clf_params = dict()
    clfs_results = dict()

    clfs['LR'] = LogisticRegression(random_state=RANDOM_STATE)
    clf_params['LR'] = {'C': np.logspace(-5, 8, 15), 'penalty':['l1', 'l2'], 'class_weight':[None, 'balanced']}
    clfs_results['LR'] = dict()

    clfs['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf_params['DT'] = {"max_depth": range(1,20),"max_features":range(1,30), 
                        "min_samples_leaf": range(1,10),"criterion": ["gini", "entropy"], 
                        'class_weight':[None, 'balanced']}
    clfs_results['DT'] = dict()

    clfs['RF'] = RandomForestClassifier(random_state=RANDOM_STATE)
    clf_params['RF'] = {"max_depth": range(1,20),"max_features":range(1,30), "min_samples_leaf": range(1,10),
                        "criterion": ["gini", "entropy"], 'class_weight':[None, 'balanced']}
    clfs_results['RF'] = dict()

    clfs['SVM'] = svm.SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    clf_params['SVM'] = {'C' :[0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'class_weight':[None, 'balanced']}
    clfs_results['SVM'] = dict()
    
    clfs['XGB'] = XGBClassifier(random_state=RANDOM_STATE)
    clf_params['XGB'] = {'n_estimators' : [50, 100, 150, 200], 'max_depth' : [2, 4, 6, 8]}
    clfs_results['XGB'] = dict()

    scoring = {'accuracy' : make_scorer(accuracy_score), 
               # 'precision' : make_scorer(precision_score),
               'precision' : 'precision',
               'recall' : make_scorer(recall_score), 
               'f1_score' : make_scorer(f1_score),
              'auc':make_scorer(roc_auc_score)}

    scorer = base_settings['SCORER']

    # GRID SEARCH PARAMETER TUNNING
    for clf in clfs.keys():  
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

            clfs_results[clf][i] = utils.get_grid_results(grid, scorer, X_test_o, y_test_o)
            r = clfs_results[clf][i]
            view(i, r)
        print('\n')


    #CRENDO CARPTAS QUE ALMACENAN LOS MODELOS
    if not os.path.exists('{}//Models'.format(FOLDER_RESULTS)):
        os.mkdir('{}//Models'.format(FOLDER_RESULTS))
        print('CREATED {}//Models'.format(FOLDER_RESULTS))
    if not os.path.exists('{}//Models//{}//'.format(FOLDER_RESULTS, scorer)):
        os.mkdir('{}//Models//{}//'.format(FOLDER_RESULTS, scorer))
        print('CREATED {}//Models//{}//'.format(FOLDER_RESULTS, scorer))

    ## SAVE MODELS
    for clf in clfs_results.keys():
        for i in clfs_results[clf].keys():
            print("Saving Classifier: {} CV {}".format(clf,i))
            joblib.dump(clfs_results[clf][i]['best_model'], 
                '{}//Models//{}//{}CV{}_{}_{}.joblib'.format(FOLDER_RESULTS, scorer, 
                    clf, i, TARGET_PREDICT, USE_PREDICTORS))
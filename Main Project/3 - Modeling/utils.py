import scikitplot as skplt
from sklearn import metrics
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
    precision_recall_fscore_support,
    make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler

import copy 
import json
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import interp
import itertools

from settings import base_settings


# py.sign_in(api_key='vM02r1sRNCokCK2O04A3', username='jcardenas.lie')
# py.sign_in(api_key='GzJkEHgbGPp7aEmHy0BZ', username='japinoza')

color_sequence = ['#f44242', '#f4eb41', '#acf441', '#285919', '#41f4eb',
                  '#4146f4', '#7041f4', '#593518', '#f441f1', '#f44173']

def save_readme():    
    exDict = {'readme': readme}
    with open('..\\results\\model_results_dataset{}\\{}\\readme_{}.txt'.format(dataset_nro, DS_NAME, DS_NAME), 'w') as file:
        file.write(str(exDict))

def save_results(results, file_name='grid_results'):
    results_copy = copy.deepcopy(results)
    for clf in results_copy.keys():
        folder = folder_results + '\\models_persistance\\{}\\'.format(clf)

        if not os.path.exists(folder):
            os.mkdir(folder)

        for cv in results_copy[clf].keys():
            for key in results_copy[clf][cv]:
                if key == 'best_model':
                    to_save = results_copy[clf][cv][key]
                    joblib.dump(to_save,'{}{}_compra_model.joblib'.format(folder, clf))
                    results_copy[clf][cv][key] = 'saved'
                if key == 'grid_cvresults':
                    pd.DataFrame(results_copy[clf][cv]['grid_cvresults']).to_excel('{}{}_{}_grid_negocio.xlsx'.format(folder, clf, cv))
                    results_copy[clf][cv]['grid_cvresults'] = 'saved'


    with open('{}.json'.format(file_name), 'w') as fp:
        json.dump(results_copy, fp)

def view_grid_results(results):
    for clf in results.keys():
        print("Classifier: ", clf)
        for cv in results[clf].keys():
                r = results[clf][cv]
                print(
            'cv: ', i,
            'acc %.4f' % r['acc'],
            '+- %.4f'% r['acc-std'],
            '| pre %.4f'% r['pre'],
            '+- %.4f'% r['pre-std'],
            '| rec %.4f'% r['rec'],
            '+- %.4f'% r['rec-std'],
            '| fs %.4f'% r['fs'],
            '+- %.4f'% r['fs-std'],
            '| auc %.4f'% r['auc'],
            '+- %.4f'% r['auc-std'],
             )  

def get_grid_results(grid, scorer, X_test_o, y_test_o, plot=False):
    best_model = grid.best_estimator_
    # grid_results = pd.DataFrame(grid.cv_results_)
    # print(grid_results.columns)
    grid_results = pd.DataFrame(grid.cv_results_)[[
    'mean_train_accuracy','mean_train_auc', 
    'mean_train_f1_score', 'mean_train_precision',
       'mean_train_recall','mean_test_accuracy', 'mean_test_auc', 
       'mean_test_f1_score', 'mean_test_precision',
       'mean_test_recall','std_test_accuracy', 'std_test_auc', 
       'std_test_f1_score', 'std_test_precision', 
       'std_test_recall', 'std_train_accuracy', 'std_train_auc', 
    'std_train_f1_score', 'std_train_precision', 
    'std_train_recall','rank_test_accuracy', 'rank_test_auc', 
    'rank_test_f1_score', 'rank_test_precision', 
    'rank_test_recall', 'params']]
    
    grid_results = grid_results.sort_values(by='rank_test_{}'.format(scorer))
    
    matrix = metrics.confusion_matrix(y_true=y_test_o, y_pred=best_model.predict(X_test_o))
    tn, fp, fn, tp = matrix.ravel()
    
    print('Validation set:')
    print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    a = (tp + tn) / (tn + fp + fn + tp)
    p = tp /(tp + fn)
    r = tp / (tp + fp)
    f = (2* p* r) / (p + r)
    auc = roc_auc_score(y_test_o, best_model.predict(X_test_o) )
    print('tpr: %.3f' % r, 'fpr %.3f' % (fp /(fp + tn)), 'auc %.2f '%auc)
    print('a: %.3f' % a, 'p %.3f' % p, 'r %.3f' % r, 'f %.3f'% f )
    print('TN %.2f'%(tn/(tn+fp)), 'TP %.2f'% p)
    
    return {
        'ts_acc':grid_results.loc[0,'mean_test_accuracy'],
        'ts_acc-std':grid_results.loc[0,'std_test_accuracy'],
        'ts_pre':grid_results.loc[0,'mean_test_precision'],
        'ts_pre-std':grid_results.loc[0,'std_test_precision'],
        'ts_rec':grid_results.loc[0,'mean_test_recall'],
        'ts_rec-std':grid_results.loc[0,'std_test_recall'],
        'ts_fs':grid_results.loc[0,'mean_test_f1_score'],
        'ts_fs-std':grid_results.loc[0,'std_test_f1_score'],
        'ts_auc':grid_results.loc[0,'mean_test_auc'],
        'ts_auc-std':grid_results.loc[0,'std_test_auc'],
        'tr_acc':grid_results.loc[0,'mean_train_accuracy'],
        'tr_acc-std':grid_results.loc[0,'std_train_accuracy'],
        'tr_pre':grid_results.loc[0,'mean_train_precision'],
        'tr_pre-std':grid_results.loc[0,'std_train_precision'],
        'tr_rec':grid_results.loc[0,'mean_train_recall'],
        'tr_rec-std':grid_results.loc[0,'std_train_recall'],
        'tr_fs':grid_results.loc[0,'mean_train_f1_score'],
        'tr_fs-std':grid_results.loc[0,'std_train_f1_score'],
        'tr_auc':grid_results.loc[0,'mean_test_auc'],
        'tr_auc-std':grid_results.loc[0,'std_test_auc'],
        'best_model': grid.best_estimator_,
        'grid_cvresults': grid.cv_results_,
    }

def view_model_results(r):
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

def plot_classifier_insights(X_test, y_test, model, col_predictors,
    folder='', model_name='Model'):
    
    clf_probas = model.predict_proba(X_test)

    #GAIN
    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_cumulative_gain(y_test, clf_probas,
        figsize=(4,3), title="{} Gain Curve".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig('{}//{}_gain_curve.png'.format(folder, model_name), 
        bbox_inches = 'tight', dpi=300)
    plt.close()
    # plt.show()

    #LIFT
    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_lift_curve(y_test, clf_probas, figsize=(4,3)\
                                                  , title="{} Lift Curve".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig('{}//{}_lift_curve.png'.format(folder, model_name),
        bbox_inches = 'tight', dpi=300)
    plt.close()
    # plt.show()
    
    # FEATURE IMPORTANCE
    
        # model.feature_importances_
    
    try:
        fig = plt.figure()
        print('TRYING FEATURE IMPORTANCE')
        ax = plt.subplot(
            skplt.estimators.plot_feature_importances(
                model, feature_names=col_predictors, x_tick_rotation=90, figsize=(4,3), 
                title="{} Feature Importance".format(model_name)))
        plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
        plt.savefig('{}//{}_f_importance.png'.format(folder, model_name),
            bbox_inches = 'tight', dpi=300)
        plt.close()
        # plt.show()

    except:
        print('FEATUE IMPORTANCE NOT SUPPORTED')
        pass
    
    #PRECISION RECALL
    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_precision_recall(y_test, clf_probas, figsize=(4,3), 
        title="{} Precision Recall".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig('{}//{}_pre_rec_curve.png'.format(folder, model_name), 
        bbox_inches = 'tight', dpi=300)
    plt.close()
    # plt.show()

    #KS STATISTICS
    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_ks_statistic(y_test, clf_probas, figsize=(4,3)\
                                                    , title="{} KS Statistics".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig('{}//{}_ks_statistics.png'.format(folder, model_name),
        bbox_inches = 'tight', dpi=300)
    plt.close()# plt.show()

def plot_all_auc_values():   
    d = dict()
    for key in roc_curves_to_plot[DS_NAME]:
        d[key] = roc_curves_to_plot[DS_NAME][key]['auc']
    
    s = [(k, d[k]) for k in sorted(d, key=d.get, reverse=True)]
    
    x_model = [x[0] for x in s]
    x_value = [x[1] for x in s]
    
    x = np.arange(len(roc_curves_to_plot[DS_NAME].keys()))
    
    f = plt.figure(figsize=(12,8))
    plt.title('{} AUC Models'.format(DS_NAME))
    plt.xlabel('Models')
    plt.ylabel('AUC')
    plt.bar(x, x_value)
    plt.xticks(x, x_model)
    # plt.savefig('..\\results\\model_results_dataset{}\\{}\\auc_models.png'.format(dataset_nro, DS_NAME,DS_NAME), dpi=300)
    plt.show()

def plot_all_roc_curves():
    
    f = plt.figure(figsize=(12,8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    for key in roc_curves_to_plot[DS_NAME]:
        plt.plot(roc_curves_to_plot[DS_NAME][key]['fpr'], roc_curves_to_plot[DS_NAME][key]['tpr']) #fpr and tpr
    
    legends = ['random choice']
    legends.extend(roc_curves_to_plot[DS_NAME].keys())
    plt.legend(legends)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Models ROC Curve'.format(DS_NAME))
    # plt.savefig('..\\results\\model_results_dataset{}\\{}\\roc_curve_all.png'.format(dataset_nro, DS_NAME, MODEL), dpi=300)
    plt.show()

def plot_battle_roc(clf, dr):
    f = plt.figure(figsize=(8,6))
    plt.plot([0, 1], [0, 1], 'k--')
    rank= 0
    legends = ['random choice']
    for cv, value in dr[clf].items():
        model =dr[clf][cv]['best_model']
        model.fit(X_grid,y_grid)
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, treshold = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        legends.append('cv %i %.4f' %(cv, roc_auc))
        plt.plot(fpr, tpr, color=color_sequence[rank]) #fpr and tpr
        rank +=1
    plt.legend(legends)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} Models ROC Curve'.format(DS_NAME))
    # plt.savefig('..\\results\\model_results_dataset{}\\{}\\{}_battle_roc_curve.png'.format(dataset_nro, DS_NAME, clf), dpi=300)
    plt.show()

def plot_battle_cv(clf, results):
    data, cv_a, acc, rec, pre, auc, fs = [], [], [], [], [], [], []
    for cv in results[clf].keys():
        r = results[clf][cv]
        cv_a.append(cv);acc.append(r['acc']);rec.append(r['rec']);pre.append(r['pre'])
        auc.append(r['auc']);fs.append(r['fs'])
    metrics = [acc,rec,pre,auc,fs]
    metrics_name = ['acc','rec','pre','auc','fs']
    for i in range(len(metrics)):
        trace = go.Scatter(
            x = cv_a,
            y = metrics[i],
            name= metrics_name[i]
        )
        data.append(trace)
    return data

def plot_box(results, metric):
    data = []

    for clf in results.keys():
        y = []
        for cv in results[clf].keys():
            y.append(results[clf][cv][metric])
        data.append(go.Box(
            y=y,
            name=clf
        ))
    layout = go.Layout(
        title = "Box Plot {}".format(metric)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_battle_conf_matrix(clf, results):
    for cv in results[clf].keys():
        model =results[clf][cv]['best_model']
    #     model.set_params(class_weight='balanced')
        model.fit(X_grid,y_grid)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(tp, tn)
        print(classification_report(y_test, y_pred))
        print('Test: Recall: %.3f' % (tp/(tp+fp)), ' Precision: %.3f' % (tp/(tp+fn)))
        # predict_mine = np.where(y_proba > 0.989, 1, 0)
    #     print(confusion_matrix(y_test, y_pred))
    #     roc_curve_plot(model, X_test, y_test)
        plot_confusion_matrix(y_test, y_pred, ['Negocio', 'No Negocio'], \
                              normalize=True, title='Confusion Matrix {} cv:{}'.format(clf, cv))

def cumulative_gain_curve_m(y_true, y_score, pos_label=None):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1] # sorting
    y_true = y_true[sorted_indices] # y_true en sorted order
    gains = np.cumsum(y_true) # 

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains

def plot_cumulative_gain_m(model, X_test, y_test, threshold=None):
    y_true = y_test

    predict_proba = model.predict_proba(X_test)
    
    if threshold:
        predict_mine = np.where(predict_probabilities > threshold, 1, 0)
        predict_proba = np.array(predict_mine)

    y_true = np.array(y_true)
    y_probas = predict_proba

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])
    #percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 1], classes[1])

    trace0 = go.Scatter(
        x = percentages,
        y = gains1,
        mode = 'lines+markers',
        name = 'class1'
    )
    
   
    trace2 = go.Scatter(
        x = [0,1],
        y = [0,1],
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4,
            dash = 'dot'
        ),
        name = 'baseline'
    )
    
    data = [trace0, trace2]
    
    layout = dict(title = f'Cumulative Gains Curve',
              xaxis = dict(title = 'Porcentaje de la Muestra (%)'),
              yaxis = dict(title = 'Gain'),
              )

    figure = go.Figure(data=data, layout=layout)

    return figure

def plot_lift_curve_m(clf_name, results, X_test, y_test, threshold = None):
    y_true = y_test
    data = []
    for cv in results[clf_name].keys():
        r = results[clf_name][cv]
        y_probas = model.predict_proba(X_test)
    
        if threshold:
            y_pred = np.where(predict_probabilities > threshold, 1, 0)

        y_true = np.array(y_true)

        classes = np.unique(y_true)
        if len(classes) != 2:
            raise ValueError('Cannot calculate Lift Curve for data with '
                             '{} category/ies'.format(len(classes)))

        # Compute Cumulative Gain Curves
        percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])

        percentages = percentages[1:]
        gains2 = gains2[1:]

        gains2 = gains2 / percentages

        trace1 = go.Scatter(
            x = percentages,
            y = gains2,
            mode = 'lines+markers',
            name = clf
        )
        data.append(trace1)
    
    trace2 = go.Scatter(
        x = [0,1],
        y = [1,1],
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4,
            dash = 'dot'
        ),
        name = 'baseline'
    )
    data.append(trace2)
    
    layout = dict(title = 'Lift Curve',
              xaxis = dict(title = 'Porcentaje de la Muestra (%)'),
              yaxis = dict(title = 'Gain'),
              )
    return {'data':data, 'layout':layout}

def plot_lift_curve(model_name, model, X_test, y_test, threshold = None):
    y_true = y_test
    data = []
    y_probas = model.predict_proba(X_test)
    
    if threshold:
        y_pred = np.where(predict_probabilities > threshold, 1, 0)

    y_true = np.array(y_true)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Lift Curve for data with '
                             '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])

    percentages = percentages[1:]
    gains2 = gains2[1:]

    gains2 = gains2 / percentages

    trace1 = go.Scatter(
        x = percentages,
        y = gains2,
        mode = 'lines+markers',
        name = model_name
    )
    data.append(trace1)
    
    trace2 = go.Scatter(
        x = [0,1],
        y = [1,1],
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4,
            dash = 'dot'
        ),
        name = 'baseline'
    )
    data.append(trace2)
    
    layout = dict(title = '{} Lift Curve'.format(model_name),
              xaxis = dict(title = 'Porcentaje de la Muestra (%)'),
              yaxis = dict(title = 'Lift'),
              )
    return {'data':data, 'layout':layout}

def plot_gain_curve(model_name, model, X_test, y_test, threshold = None):
    y_true = y_test

    predict_proba = model.predict_proba(X_test)
    
    if threshold:
        predict_mine = np.where(predict_probabilities > threshold, 1, 0)
        predict_proba = np.array(predict_mine)

    y_true = np.array(y_true)
    y_probas = predict_proba

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])
    #percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 1], classes[1])

    trace0 = go.Scatter(
        x = percentages,
        y = gains1,
        mode = 'lines+markers',
        name = 'class1'
    )

    #trace1 = go.Scatter(
    #   x = percentages,
    #   y = gains2,
    #   mode = 'lines+markers',
    #   name = 'class2'
    #)
    
   
    trace2 = go.Scatter(
        x = [0,1],
        y = [0,1],
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4,
            dash = 'dot'
        ),
        name = 'baseline'
    )
    
    data = [trace0, trace2]
    
    layout = dict(title = '{} Cumulative Gains Curve'.format(model_name),
              xaxis = dict(title = 'Porcentaje de la Muestra (%)'),
              yaxis = dict(title = 'Gain'),
              )

    figure = go.Figure(data=data, layout=layout)

    return figure

# FEATURE IMPORTANCE
def plot_importance_graph(model):
    features = personas.columns
    
    feature_importance = model.best_estimator_.feature_importances_ 
    fig = plt.figure(figsize=(20, 18))
    ax = fig.add_subplot(111)

    df_f = pd.DataFrame(feature_importance, columns=["importance"])
    df_f["labels"] = features
    df_f.sort_values("importance", inplace=True, ascending=False)
    display(df_f.head(5))

    index = np.arange(len(feature_importance[:20]))
    bar_width = 0.5
    rects = plt.barh(index[:20] , df_f["importance"][:20], bar_width, alpha=0.4, color='b', label='Main')
    plt.yticks(index, df_f["labels"])
    plt.title("{} {} Feature Importance".format(DS_NAME, MODEL))
    plt.savefig('..\\results\\model_results_dataset{}\\{}\\{}_importance.png'.format(dataset_nro, DS_NAME, MODEL), dpi=300)
    plt.show()

    # Import necessary modules

def plot_combo_chart(feature, model, dataset, target, title='Casos Éxito', grupo=None):
    if grupo == None:
        grupo = feature
    
    dataset['1'] = 1
    data = dict()
    personas_view = copy.deepcopy(dataset)
    columna = feature
   
    # print(target, 'no {}'.format(target))
    for group, frame in personas_view.groupby(columna):
        target_sum = frame[frame[target] == True]['1'].sum()
        target_no_sum = frame[frame[target] == False]['1'].sum()
        data[group] = {}
        data[group][target] = target_sum
        data[group]['no {}'.format(target)] = target_no_sum
        # data[group][target] = 
        # data[group]['no conversión'] = 

    to_x, to_neg, to_noneg = [], [], []
    
    for key in data.keys():
        to_x.append('{}'.format(key))
        to_neg.append(data[key][target])
        to_noneg.append(data[key]['no {}'.format(target)])
    mult = int(len(to_x)/5)

    to_g_name, to_g_neg, to_g_noneg = [], [], []
    
    max_range = 5
    if personas_view[feature].dtype != bool:
        for i in range(max_range):
            name = '{} - {}'.format(to_x[mult*i], to_x[mult*(i+1)])
            to_g_name.append(name)
            to_g_neg.append(sum(to_neg[(mult*i):(mult*(i+1))]))
            to_g_noneg.append(sum(to_noneg[(mult*i):(mult*(i+1))]))
    else:
        to_g_name = ['Falso', 'Verdadero']
        to_g_neg = to_neg
        to_g_noneg = to_noneg
    
    exito = [float("{0:.2f}".format(x/(x+y))) for x, y in zip(to_g_neg, to_g_noneg)]

    trace1 = go.Bar(
        x=to_g_name,
        y=to_g_neg,
        name=target,
        marker=dict(
            color='rgb(55, 83, 109)'
        )
    )
    trace2 = go.Bar(
        x=to_g_name,
        y=to_g_noneg,
        name='No {}'.format(target),
        marker=dict(
            color='rgb(26, 118, 255)'
        )
    )

    trace3 = go.Scatter(
        x=to_g_name,
        y=exito,
        yaxis='y2',
        name='% Éxito'
    )
    traces = [trace1, trace2, trace3]

    layout = go.Layout(
        title=title,
    #font = dict(size=24),
        xaxis=dict(
            title = grupo,
            titlefont=dict(size=18,color='rgb(0, 0, 0)'),
            tickfont=dict(
                size=14,
                color='rgb(0, 0, 0)'
            )
        ),
        yaxis=dict(
            title='Número Personas',
            titlefont=dict(
                size=18,
                color='rgb(0, 0, 0)'
            ),
            tickfont=dict(
                size=18,
                color='rgb(0, 0, 0)'
            )
        ),
        yaxis2=dict(
            title='% Éxito',
            titlefont=dict(
                size=18,
                color='rgb(0, 0, 0)'
            ),
            tickfont=dict(
                size=18,
                color='rgb(0, 0, 0)'
            ),
            overlaying='y',
            side='right'
        ),
        legend=dict(
            x=1.1,
            y=1.0,
            font=dict(size=14),
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='stack',
        bargap=0.15,
        bargroupgap=0.5
    )
    figure = {'data':traces, 'layout':layout}
    return figure

def plot_confusion_matrix(y, y_pred, classes, folder, model_name,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm = np.array([[tp,fn],[fp,tn]])
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    if normalize:
        save_name = '{}//conf_matrix_normalize.png'.format(folder, model_name)
    else:
        save_name = '{}//conf_matrix.png'.format(folder, model_name)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    # plt.show()

def plot_roc_curve(model, X_test, y_test, folder='', title='Model ROC Curve' ): 
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = model.predict_proba(X_test)[:,1]
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, treshold = roc_curve(y_test, y_pred_prob)
    
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=model.predict(X_test))
    tn, fp, fn, tp = matrix.ravel()
    fpr_point = fp / (fp + tn)
    tpr_point = tp / (tp + fp)
    
    print('treshold: ',treshold)
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("AUC: ",roc_auc_score(y_test, y_pred_prob))
    print("Point in ROC fpr:{}, tpr:{}".format(fpr_point,tpr_point))

    predict_mine = np.where(y_pred_prob > 0.21, 1, 0)
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=model.predict(X_test))
    tn, fp, fn, tp = matrix.ravel()
    fpr_point = fp / (fp + tn)
    tpr_point = tp / (tp + fp)
    
    print("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    print("AUC: ",roc_auc_score(y_test, y_pred_prob))
    print("Point in ROC fpr:{}, tpr:{}".format(fpr_point, tpr_point))
    # Plot ROC curve
    #plt.plot(fpr_point,tpr_point)
    markers_on = [fpr_point]
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False  Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.savefig(folder + '//{}.png', dpi=300)
    plt.close()
    # plt.show()
    
def cross_roc_validation(model, model_name, folder, X_cros, y_cros, X_val, y_val, splits=3):
    
    kf = StratifiedKFold(n_splits=splits)
    classifier = model

    tprs = []
    aucs, accs, pres, recs, fs = [], [], [], [],[]
    mean_fpr = np.linspace(0, 1, 100)
    r_tpr, r_fpr = [],[]
    i = 0
    X_val, y_val = X_val, y_val
    for train, test in kf.split(X_cros, y_cros):
        X_train, y_train, X_test, y_test  = X_cros.iloc[train], y_cros.iloc[train], X_cros.iloc[test], y_cros.iloc[test]
        
        probas_ = classifier.fit(X_train, y_train ).predict_proba(X_test)
        y_pred=classifier.predict(X_test)
        
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        r_fpr.append(fpr)
        r_tpr.append(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)
        tn, fp, fn, tp = matrix.ravel()

        a = (tp + tn) / (tn + fp + fn + tp)
        p = tp /(tp + fp)
        r = tp / (tp + fn)
        f = (2* p* r) / (p + r)
        accs.append(a); pres.append(p); recs.append(r); fs.append(f)
        auc_score = roc_auc_score(y_test, classifier.predict(X_test))
        i += 1

    results = {
        'model' : model_name,
        'CV Splits': splits,
        'acc': " %.3f +- %.3f"%(np.mean(accs),np.std(accs)),
        'pre':" %.3f +- %.3f"%(np.mean(pres),np.std(pres)),
        'rec': "%.3f +- %.3f"%(np.mean(recs),np.std(recs)),
        'fs': "%.3f +- %.3f"%(np.mean(fs),np.std(fs)),
        'auc': "%.3f +- %.3f"%(np.mean(aucs),np.std(aucs))
    }

    y_val_probas = classifier.fit(X_cros, y_cros).predict_proba(X_val)
    y_val_pred=classifier.predict(X_val)

    matrix = metrics.confusion_matrix(y_true=y_val, y_pred=y_val_pred)
    tn, fp, fn, tp = matrix.ravel()

    print('Validation set:')
    print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)
    a = (tp + tn) / (tn + fp + fn + tp)
    p = tp /(tp + fp)
    r = tp / (tp + fn)
    f = (2* p* r) / (p + r)
    print('tpr: %.3f' % r, 'fpr %.3f' % (fp /(fp + tn)), 'auc %.2f '%auc_score)
    print('a: %.3f' % a, 'p %.3f' % p, 'r %.3f' % r, 'f %.3f'% f )
    print('TN %.2f'%(tn/(tn+fp)), 'TP %.2f'% p)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC Curve'.format(model_name))
    plt.legend(loc="lower right",  bbox_to_anchor=(1.7, 0))
    plt.savefig('{}//cross_roc_val.png'.format(folder, model_name), 
        bbox_inches = 'tight', dpi=300)
    plt.close()
    # plt.show()
    
    return mean_fpr, mean_tpr, results

def cross_lift_validation(model, model_name, folder, X_cros, y_cros, X_val, y_val, splits=3):
    kf = StratifiedKFold(n_splits=splits)
    classifier = model
    
    tprs = []
    gains = []
    percentage_out = None
    size_out, i  = 0, 0
    aucs, accs, pres, recs, fs = [], [], [], [],[]
    mean_fpr = np.linspace(0, 1, 100)
    r_tpr, r_fpr = [],[]
    X_val, y_val = X_val, y_val
    for train, test in kf.split(X_cros, y_cros):
        X_train, y_train, X_test, y_test  = X_cros.iloc[train], y_cros.iloc[train], X_cros.iloc[test], y_cros.iloc[test]
        probas_ = classifier.fit(X_train, y_train ).predict_proba(X_test)
        y_pred=classifier.predict(X_test)
        
        if i == 0: size_out = y_test.shape[0]; i+=1
                
        # Compute Lift
        y_true = y_test
        y_probas = probas_
        y_true = np.array(y_true)
        classes = np.unique(y_true)
    
        ## Compute Cumulative Gain Curves
        percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])
        percentages = percentages[1:]
        gains2 = gains2[1:]
        gains2 = gains2 / percentages
        if gains2.shape[0] < size_out: 
            gains2 = np.insert(gains2, -1, 1)
            percentages = np.insert(percentages, -1, 1)
        percentages_out = percentages
        
        ################################################################################################

        gains.append(gains2)
        plt.plot(percentages, gains2, lw=3, alpha=0.3)
    
    plt.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    plt.xlabel('Percentage of sample', fontsize="medium")
    plt.ylabel('Lift', fontsize="medium")
    plt.tick_params(labelsize="medium")
    plt.grid('on')
    plt.legend(loc='lower right', fontsize="medium")
    plt.plot()

    # MEAN GAIN
    mean_gains = np.mean(gains, axis=0)
    mean_gains[-1] = 1.0
    plt.plot(percentages, mean_gains, lw=3, color='b')
    plt.savefig('{}//cross_lift_val.png'.format(folder, model_name), bbox_inches = 'tight', dpi=300)
    plt.close()
    
    return mean_gains, percentages_out

def cross_lift50_validation(model, model_name, folder, X_cros, y_cros, splits=3):
    kf = StratifiedKFold(n_splits=splits)
    classifier = model
    
    gains = []
    percentage_out = None
    size_out, i  = 0, 0
    for train, test in kf.split(X_cros, y_cros):
        X_train, y_train, X_test, y_test  = X_cros.iloc[train], y_cros.iloc[train], X_cros.iloc[test], y_cros.iloc[test]
        
        smote = SMOTE(ratio='minority')
        tl = TomekLinks(return_indices=True, ratio='majority')
        rus = RandomUnderSampler(return_indices=True)
        ros = RandomOverSampler()
        
        X_sm, y_sm = ros.fit_sample(X_test, y_test)
        # X_sm, y_sm, id_sm = tl.fit_sample(X_test, y_test)
        # X_sm, y_sm, id_rus = rus.fit_sample(X_test, y_test)
        # X_sm, y_sm = smote.fit_sample(X_test, y_test)
        X_sm = pd.DataFrame(X_sm)
        X_sm.columns = X_train.columns
             
        probas_ = classifier.fit(X_train, y_train ).predict_proba(X_sm)
        y_pred=classifier.predict(X_sm)
        
        if i == 0: size_out = y_sm.shape[0]; i+=1
                
        # Compute Lift
        y_true = y_sm
        y_probas = probas_
        y_true = np.array(y_true)
        classes = np.unique(y_true)
    
        ## Compute Cumulative Gain Curves
        percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 0], classes[0])
        percentages = percentages[1:]
        gains2 = gains2[1:]
        gains2 = gains2 / percentages
        # print(gains2[:20])
        if gains2.shape[0] < size_out:
            size = gains2.shape[0]
            while size < size_out:
                size +=1
                gains2 = np.insert(gains2, -1, 1)
                percentages = np.insert(percentages, -1, 1)
        print(len(gains2))
        percentages_out = percentages
        
        ################################################################################################
        gains.append(gains2)
        plt.plot(percentages, gains2, lw=3, alpha=0.3)
    
    plt.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')

    plt.xlabel('Percentage of sample', fontsize="medium")
    plt.ylabel('Lift', fontsize="medium")
    plt.tick_params(labelsize="medium")
    plt.grid('on')
    plt.legend(loc='lower right', fontsize="medium")
    plt.plot()


    # MEAN GAIN
    mean_gains = np.mean(gains, axis=0)
    mean_gains[-1] = 1.0
    plt.plot(percentages, mean_gains, lw=3, color='b')
    plt.savefig('{}//cross_lift50_val.png'.format(folder, model_name), bbox_inches = 'tight', dpi=300)
    # plt.show()
    plt.close()
    return mean_gains, percentages_out
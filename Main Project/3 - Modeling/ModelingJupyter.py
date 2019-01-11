
# coding: utf-8

# # 0 Utils

# In[1]:


import os
import numpy as np
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
# sb.set()
import matplotlib.pyplot as plt
import pandas as pd
import shap
import copy


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


N_JOBS = 6 # -1 ocupa todo los cores
RANDOM_STATE = 42 # para que sea reproducible


# # 1 Predictors

# In[4]:


predictors_set = dict()

no_is_time_price = [
    'is_recontacto', 'is_remoto', 'is_descuento', 'valid_rut',
    'loc_comuna', 'loc_provincia', 'loc_region', 'sexo', 'tipo_cliente',
    'mean_cot_bod',
    'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 'medio_inicial',
    'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta',
    'nro_cot_estu', 'nro_proyectos', 'precio_cotizacion_media',
    'precio_cotizacion_median', 'precio_cotizacion_std', 
    
    'tiempo_cotizacion_media', 'tiempo_cotizacion_median',
    'tiempo_cotizacion_std',   
    'Altos del Valle',
    'Edificio Urban 1470', 
#     'San Andres Del Valle', 
    'Edificio Mil610',
       'Edificio Junge']

predictors_set ['nois'] = no_is_time_price

is_no_time_price = ['actividad', 'is_apellido1', 'is_apellido2',
       'is_celular', 'is_direccion', 'is_fnac', 'is_nombre',
       'is_nombrecompleto', 'is_nrofam', 'is_presencial', 'is_profesion',
       'is_recontacto', 'is_remoto', 'is_telefono', 'loc_comuna',
       'loc_provincia', 'loc_region', 'medio_inicial', 'sexo']

predictors_set ['isno'] = is_no_time_price 

complete = [
    'actividad', 'is_apellido1', 'is_apellido2',
       'is_celular', 'is_descuento', 'is_direccion', 'is_fnac', 'is_nombre',
       'is_nombrecompleto', 'is_nrofam', 'is_presencial', 'is_profesion',
       'is_recontacto', 'is_remoto', 'is_telefono', 'loc_comuna',
       'loc_provincia', 'loc_region', 'max_rango_edad', 'medio_inicial', 'sexo'
    
    'mean_cot_bod',
    'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 'medio_inicial',
    'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta',
    'nro_cot_estu', 'nro_proyectos', 'precio_cotizacion_media',
    'precio_cotizacion_median', 'precio_cotizacion_std', 
    
    'tiempo_cotizacion_media', 'tiempo_cotizacion_median',
    'tiempo_cotizacion_std',   
    'Altos del Valle',
    'Edificio Urban 1470', 
#     'San Andres Del Valle', 
    'Edificio Mil610',
       'Edificio Junge']

predictors_set [''] = complete 


# In[5]:


personas10 = pd.read_csv('..\\..\\Datos\\experiments\\personas_cotizacion10.csv' , index_col=[0], encoding = "ISO-8859-1")
personas8 = pd.read_csv('..\\..\\Datos\\experiments\\personas_cotizacion8.csv' , index_col=[0], encoding = "ISO-8859-1")


# In[6]:


s1 = set(personas10.columns)
s2 = set(personas8.columns)
matched = s1.intersection(s2) # set(['dog', 'cat', 'donkey'])
unmatched = s1.symmetric_difference(s2) # set(['pig'])
unmatched


# # 1 Set Experiment

# In[7]:


base_folder = 'resultados'
if not os.path.exists(base_folder):
    os.mkdir(base_folder)
    
predictores = 'isno'
if not os.path.exists('{}\\{}'.format(base_folder, predictores)):
    os.mkdir('{}\\{}'.format(base_folder, predictores))

target_variable = 'negocio'
if not os.path.exists('{}\\{}\\{}'.format(base_folder, predictores, target_variable)):
    os.mkdir('{}\\{}\\{}'.format(base_folder, predictores, target_variable))


# In[8]:


folder_results = '{}\\{}\\{}'.format(base_folder, predictores, target_variable)


# In[9]:


# dataset, base nro 2
dataset_nro = 10
experiment_nro = 1
# 'All Proyects', 'Altos del Valle', 'Edificio Urban 1470','San Andres Del Valle', 'Edificio Mil610', 'Edificio Junge'
proyecto_select = 'san andres del valle' #cambiable

# base, base_profesion, base_minusisprofesion, base_medini_isrec
use_predictors = 'isno' #cambiable
# experimento 2 y 4 va con True
no_nulls = False #cambiable
sesgo = False #cambiable


# In[10]:


# nombre experimento
exp_name = "{} {}".format(proyecto_select, use_predictors)
# eleccion dataset
dataset = 'personas_cotizacion{}.csv'.format(dataset_nro)

DS_NAME = exp_name


# # 1 Load Dataset

# In[11]:


personas = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(dataset), index_col=[0], encoding = "ISO-8859-1")
personas_info = pd.read_csv('..\\..\\Datos\\experiments\\personas_cotizacion10.csv', index_col=[0], encoding = "ISO-8859-1")

print(personas.shape)
print(personas_info.shape)


# In[12]:


personas.columns


# In[13]:


# Se pasan las variables categoricas que son objetos a variables categoricas
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


# In[14]:


# Personas que solo cotizaron en un proyecto especifico
# Para E5D5, E6D5 #########################################
if proyecto_select != 'All Proyects':
    mask = (personas[proyecto_select] > 0)
    personas = personas[mask]
    
    mask = (personas_info[proyecto_select] > 0)
    personas_info = personas_info[mask]

    ###########################################################
print(personas.shape)
print(personas_info.shape)
predictors = predictors_set[use_predictors]

if no_nulls:
    personas.replace(['sin informacion'], np.nan, inplace=True)
    personas.dropna(inplace=True)


# In[15]:


# personas.to_excel('personas_filtro.xlsx')
# personas.to_csv('personas_filtro.csv')


# In[16]:


personas_original = copy.deepcopy(personas)
personas_original.shape


# # DEF VARIABLE OBJETIVO

# In[17]:


# Compra como variable objetivo
p_target = pd.DataFrame(personas.negocio) # Target EDITABLE
p_target.head()


# In[18]:


# Cambio de [True, False] a [1, -1] para que salgan bien los resultados de la conf matrix
p_target.negocio = [1 if x == True else -1 for x in p_target.negocio] # EDITABLE

personas = personas[predictors]


# In[19]:


# personas.to_excel('tmp.xlsx')
# personas.to_csv('tmp.csv')


# In[20]:


# Corroborando que los largos de personas y sus variables objetivos sean iguales

print(p_target.shape)
print(personas.shape)
both = set(personas.index) & set(p_target.index)
print('Len', len(both))
      

print(personas.shape)


personas = pd.get_dummies(personas)
print(personas.shape)
# personas.describe()


# In[21]:


personas.columns


# # 3 Supervised Learning: Classification

# ## 3.1 Libraries

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn import svm
from xgboost import XGBClassifier

import itertools

from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ## 3.2 Check the Dataset

# ## 3.4 Help Functions

# In[23]:


import scikitplot as skplt

def classifier_insights(y_test, clf_probas, model, f_importance = False, model_name='Model'):
    
    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_cumulative_gain(y_test, clf_probas,                                                        figsize=(4,3), title="{} Gain Curve".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig(folder_results + '\\{}_cum_gain.png'.format(model_name), bbox_inches = 'tight', dpi=300)
    plt.show()


    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_lift_curve(y_test, clf_probas, figsize=(4,3)                                                  , title="{} Lift Curve".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig(folder_results + '\\{}_lift.png'.format(model_name), bbox_inches = 'tight', dpi=300)
    plt.show()
    
    if f_importance:
        fig = plt.figure()
        ax = plt.subplot(skplt.estimators.plot_feature_importances(
            model, feature_names=col_predictors, x_tick_rotation=90, figsize=(4,3)\
        , title="{} Feature Importance".format(model_name)))
        plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
        plt.savefig(folder_results + '\\{}_f_importance.png'.format(model_name), bbox_inches = 'tight', dpi=300)
        plt.show()

    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_precision_recall(y_test, clf_probas, figsize=(4,3)                                                        , title="{} Precision Recall".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig(folder_results + '\\{}_precition_recall.png'.format(model_name), bbox_inches = 'tight', dpi=300)
    plt.show()

    fig = plt.figure()
    ax = plt.subplot(skplt.metrics.plot_ks_statistic(y_test, clf_probas, figsize=(4,3)                                                    , title="{} KS Statistics".format(model_name)))
    plt.legend(loc=9, bbox_to_anchor=(1.15, 1), ncol=1)
    plt.savefig(folder_results + '\\{}_ks_sta.png'.format(model_name), bbox_inches = 'tight', dpi=300)
    plt.show()


# In[24]:


import json
def save_readme():    
    exDict = {'readme': readme}
    with open('..\\results\\model_results_dataset{}\\{}\\readme_{}.txt'.format(dataset_nro, DS_NAME, DS_NAME), 'w') as file:
        file.write(str(exDict))

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
#     plt.savefig('..\\results\\model_results_dataset{}\\{}\\auc_models.png'.format(dataset_nro, DS_NAME,DS_NAME), dpi=300)
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
#     plt.savefig('..\\results\\model_results_dataset{}\\{}\\roc_curve_all.png'.format(dataset_nro, DS_NAME, MODEL), dpi=300)
    plt.show()


# In[25]:


# SAVE RESULTS
from sklearn.externals import joblib
import copy 
import json

# results_copy = copy.deepcopy(clfs_results)

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


# In[26]:


color_sequence = ['#f44242', '#f4eb41', '#acf441', '#285919', '#41f4eb',
                  '#4146f4', '#7041f4', '#593518', '#f441f1', '#f44173']

def battle_roc(clf, dr):
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
#     plt.savefig('..\\results\\model_results_dataset{}\\{}\\{}_battle_roc_curve.png'.format(dataset_nro, DS_NAME, clf), dpi=300)
    plt.show()


# In[27]:


def battle_cv(clf, results):
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


# In[28]:


def view_results(results):
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


# In[29]:


import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in(api_key='vM02r1sRNCokCK2O04A3', username='jcardenas.lie')
# py.sign_in(api_key='GzJkEHgbGPp7aEmHy0BZ', username='japinoza')

def box_plot(results, metric):
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


# In[30]:


def battle_conf_matrix(clf, results):
    for cv in results[clf].keys():
        model =results[clf][cv]['best_model']
#         model.set_params(class_weight='balanced')
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
        plot_confusion_matrix(y_test, y_pred, ['Negocio', 'No Negocio'],                               normalize=True, title='Confusion Matrix {} cv:{}'.format(clf, cv))


# In[31]:


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

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

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
#     percentages, gains2 = cumulative_gain_curve_m(y_true, y_probas[:, 1], classes[1])

    trace0 = go.Scatter(
        x = percentages,
        y = gains1,
        mode = 'lines+markers',
        name = 'class1'
    )

#     trace1 = go.Scatter(
#         x = percentages,
#         y = gains2,
#         mode = 'lines+markers',
#         name = 'class2'
#     )
    
   
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


def plot_lift_curve_m(clf, results, X_test, y_test, threshold = None):
    y_true = y_test
    data = []
    for cv in results[clf].keys():
        r = results[clf][cv]
        
#         model.fit(X_grid, y_grid)
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


# In[32]:


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
def plot_confusion_matrix(y, y_pred, classes,
                        normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, folder=''):
    
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
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(folder_results + '{}.png'.format(title), dpi=300)
    plt.show()


def roc_curve_plot(model, X_test, y_test, folder='', title='Model ROC Curve' ):
    
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = model.predict_proba(X_test_o)[:,1]

    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, treshold = roc_curve(y_test_o, y_pred_prob)
    
    matrix = metrics.confusion_matrix(y_true=y_test_o, y_pred=model.predict(X_test_o))
    tn, fp, fn, tp = matrix.ravel()
    fpr_point = fp / (fp + tn)
    tpr_point = tp / (tp + fp)
    
    print("AUC: ",roc_auc_score(y_test_o, y_pred_prob))
    print("Point in ROC", fpr_point,tpr_point)

    # Plot ROC curve
#     plt.plot(fpr_point,tpr_point)
    markers_on = [fpr_point]
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.savefig(folder_results + '{}.png'.format(title), dpi=300)
    plt.show()
    
def grid_results(grid, scorer, plot=False):
    
    best_model = grid.best_estimator_
    
    grid_results = pd.DataFrame(grid.cv_results_)[['mean_train_accuracy',
       'mean_train_auc', 'mean_train_f1_score', 'mean_train_precision',
       'mean_train_recall','mean_test_accuracy',
       'mean_test_auc', 'mean_test_f1_score', 'mean_test_precision',
       'mean_test_recall','std_test_accuracy', 'std_test_auc', 'std_test_f1_score',
       'std_test_precision', 'std_test_recall',
        'std_train_accuracy', 'std_train_auc', 'std_train_f1_score',
       'std_train_precision', 'std_train_recall','rank_test_accuracy', 'rank_test_auc', 'rank_test_f1_score',
       'rank_test_precision', 'rank_test_recall', 'params']]
    
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


# ## 3.3 Train & Test

# In[33]:


both = set(personas.index) & set(p_target.index)
len(both)


# In[34]:


# personas.reset_index(inplace=True)
# p_negocio.reset_index(inplace=True)
# my_indices = p_negocio.index


# In[35]:


X = personas 
col_predictors = X.columns
y = p_target

#_grid para hacer cross_validation y _test_o para hacer validacion sobre datos no vistos
X_grid, X_test_o, y_grid, y_test_o = train_test_split(X, y                                                     ,test_size=0.2, random_state=RANDOM_STATE)
# X_grid, X_test_o, y_grid, y_test_o, indices_train, indices_test = train_test_split(X, y \
#                                                     , indices,test_size=0.2, random_state=RANDOM_STATE)
print(X_grid.shape, X_test_o.shape, y_grid.shape, y_test_o.shape)


X_test_o.to_excel('x_test_compra_{}.xlsx'.format(use_predictors))
y_test_o.to_excel('y_test_compra_{}.xlsx'.format(use_predictors))


# In[36]:


X_test_o.head(1)


# In[37]:


personas_original.loc[1573]


# In[38]:


# Los indices de arriba debieran ser los mismos


# ## 3.5 Models

# In[39]:


clfs = dict()
clfs['LR'] = LogisticRegression(random_state=RANDOM_STATE)
clfs['DT'] = DecisionTreeClassifier(random_state=RANDOM_STATE)
clfs['RF'] = RandomForestClassifier(random_state=RANDOM_STATE)
clfs['SVM'] = svm.SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
# clfs['KNN'] = KNeighborsClassifier()
# clfs['PCAKNN'] = Pipeline([('pca', PCA(random_state=RANDOM_STATE)), ('clf', KNeighborsClassifier())])
clfs['XGB'] = XGBClassifier(random_state=RANDOM_STATE)


# In[40]:


clf_params = dict()
# clf_params['KNN'] = {'n_neighbors': np.arange(1, 20)}
# clf_params['PCAKNN'] = {'pca__n_components': [2, 3, 4, 5, 6, 7, 8, 9]}
clf_params['LR'] = {'C': np.logspace(-5, 8, 15), 'penalty':['l1', 'l2'], 'class_weight':[None, 'balanced']}
clf_params['DT'] = {"max_depth": range(1,20),"max_features":range(1,30), 
                    "min_samples_leaf": range(1,10),"criterion": ["gini", "entropy"], 
                    'class_weight':[None, 'balanced']}
clf_params['RF'] = {"max_depth": range(1,20),"max_features":range(1,30), "min_samples_leaf": range(1,10),
                    "criterion": ["gini", "entropy"], 'class_weight':[None, 'balanced']}
clf_params['SVM'] = {'C' :[0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'class_weight':[None, 'balanced']}
clf_params['XGB'] = {'n_estimators' : [50, 100, 150, 200], 'max_depth' : [2, 4, 6, 8]}


# In[41]:


clfs_results = dict()
clfs_results['LR'] = dict()
clfs_results['DT'] = dict()
clfs_results['RF'] = dict()
# clfs_results['PCAKNN'] = dict()
# clfs_results['KNN'] = dict()
clfs_results['XGB'] = dict()
clfs_results['SVM'] = dict()


# In[42]:


def view_grid(r):
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


# # Grid

# In[43]:



scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
          'auc':make_scorer(roc_auc_score)}

for clf in clfs.keys():
    scorer = 'f1_score'
    print("Classifier:",clf, ' Scorer:', scorer)
    for i in range(3,4):

        kf = StratifiedKFold(n_splits=i, shuffle=True, random_state=RANDOM_STATE)

        if clf == 'RF' or clf == 'DT':
            grid = RandomizedSearchCV(clfs[clf], clf_params[clf],                                       cv=kf, n_jobs=N_JOBS, n_iter=300,                                       random_state=RANDOM_STATE, scoring=scoring, refit=scorer)
        else:
            grid = GridSearchCV(clfs[clf], clf_params[clf], cv=kf, n_jobs=N_JOBS                                , scoring=scoring, refit=scorer)
        
        grid.fit(X_grid, y_grid)

        clfs_results[clf][i] = grid_results(grid, scorer)
        r = clfs_results[clf][i]
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
    print('\n')
    


# In[44]:


save_results(clfs_results)


# # LIFT & GAIN

# In[ ]:


model_name = 'RF'


# In[ ]:


model = clfs_results[model_name][3]['best_model']


# In[ ]:


model


# In[ ]:


data = plot_lift_curve_m(model_name, clfs_results, X_test_o, y_test_o)
py.iplot(data, filename='basic-line')


# In[ ]:


data = plot_cumulative_gain_m(model, X_test_o, y_test_o, threshold=None)
py.iplot(data, filename='basic-line')


# # ANALYSIS: DESCRIBE ATRIBUTES

# In[ ]:


def combo_chart(feature, model, dataset, target ,title='Casos Éxito', grupo=None):
    if grupo == None:
        grupo = feature

    dataset['1'] = 1
    data = dict()
    personas_view = copy.deepcopy(dataset)
    columna = feature

    for group, frame in personas_view.groupby(columna):
        data[group] = {}
        data[group]['conversión'] = frame[frame[target] == True]['1'].sum()
        data[group]['no conversión'] = frame[frame[target] == False]['1'].sum()

    to_x, to_neg, to_noneg = [], [], []


    for key in data.keys():
        to_x.append('{}'.format(key))
        to_neg.append(data[key]['conversión'])
        to_noneg.append(data[key]['no conversión'])
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
        name='Conversión',
        marker=dict(
            color='rgb(55, 83, 109)'
        )
    )
    trace2 = go.Bar(
        x=to_g_name,
        y=to_g_noneg,
        name='No Conversión',
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
#         font = dict(size=24),
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
    figure = go.Figure(data=traces, layout=layout)
    return figure


# In[ ]:


data = combo_chart('precio_cotizacion_median', model, personas,'compra', title='Casos Éxito Compra', grupo='Grupo (Median Cotizaciones)')
py.iplot(data, filename='basic-line')


# In[52]:


print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def cross_roc_validation(model, model_name, X_cros, y_cros, X_val, y_val, splits=3):
    
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
    
    print('Test ','CV Splits', splits, '\n',
        "acc %.3f +- %.3f"%(np.mean(accs),np.std(accs)),\
          "pre %.3f +- %.3f"%(np.mean(pres),np.std(pres)), \
          "rec %.3f +- %.3f"%(np.mean(recs),np.std(recs)),\
          "fs %.3f +- %.3f"%(np.mean(fs),np.std(fs)),\
          "auc %.3f +- %.3f"%(np.mean(aucs),np.std(aucs)),)
    
    y_val_probas = classifier.fit(X, y).predict_proba(X_val)
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
    plt.savefig(folder_results + '\\{}_roc_cv_{}.png'.format(model_name, splits), bbox_inches = 'tight', dpi=300)
    plt.show()
    
    return mean_fpr, mean_tpr


# In[53]:


mean_rocs = dict()
for key in clfs_results.keys():
    print("Classifier ", key)
    model =clfs_results[key][3]['best_model']
    mean_fpr, mean_tpr = cross_roc_validation(model, key, X_grid, y_grid, X_test_o, y_test_o, splits=6)
    mean_rocs[key] = dict()
    mean_rocs[key]['fpr'] = mean_fpr
    mean_rocs[key]['tpr'] = mean_tpr


# In[ ]:


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
for key in mean_rocs.keys():
    fpr = mean_rocs[key]['fpr']
    tpr = mean_rocs[key]['tpr']
    plt.plot(fpr, tpr, lw=1, label=key)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('All models ROC Curve')
plt.legend(loc="lower right",  bbox_to_anchor=(1, 0))
plt.savefig(folder_results + '\\all_models_roc.png', bbox_inches = 'tight', dpi=300)
plt.show()


# In[54]:


model = clfs_results['RF'][3]['best_model']
from sklearn.externals import joblib
joblib.dump(model, 'RFCV3_negocio_is.joblib') 


# In[ ]:


model_name = 'XGB'
model =clfs_results[model_name][3]['best_model']
# model.fit(X_grid,y_grid)
y_pred = model.predict(X_test_o)
y_proba = model.predict_proba(X_test_o)[:,1]
# predict_mine = np.where(y_proba > 0.989, 1, 0)
print(classification_report(y_test_o, y_pred ))
print(confusion_matrix(y_test_o, y_pred))
roc_curve_plot(model, X_test_o, y_test_o,  title='Val {} Negocio ROC Curve'.format(model_name, 2))
plot_confusion_matrix(y_test_o, y_pred, ['Negocio', 'No Negocio'] ,                        title='Val {} Norm. Confusion Matrix'.format(model_name, 2), normalize=True)
plot_confusion_matrix(y_test_o, y_pred, ['Negocio', 'No Negocio'] ,                        title='Val {} Confusion Matrix'.format(model_name, 2), normalize=False)


# In[ ]:


model_name = 'XGB'
model = clfs_results[model_name][3]['best_model']
y_pred = model.predict(X_test_o)
y_probas = model.predict_proba(X_test_o)

print(X_test_o.shape, y_test_o.shape, y_proba.shape)
classifier_insights(y_test_o, y_probas, model, f_importance = True, model_name='{} Negocio'.format(model_name))


# # PREDICT ONE CLIENTE

# In[ ]:


model_name = 'RF'
probas = clfs_results[model_name][2]['best_model'].predict_proba(X_test_o)[:,1]
y_pred = clfs_results[model_name][2]['best_model'].predict(X_test_o)


# In[ ]:


print(len(probas), len(y_pred))


# In[ ]:


b_personas = copy.deepcopy(personas)
b_p_negocio = copy.deepcopy(p_target)


# In[ ]:


b_personas = b_personas.loc[X_test_o.index]
b_p_negocio = b_p_negocio.loc[X_test_o.index]


# In[ ]:


b_personas['y_pred'] = y_pred
b_personas['probas'] = probas


# In[ ]:


print(b_personas.shape)
b_personas.head(1)


# In[ ]:


# personas_original.loc[1573]


# In[ ]:


info = personas_original.loc[[1573]].merge(personas_info, left_on='rut', right_on='rut', how='left')
info[['nombre','y_pred']]


# In[ ]:


b_personas.drop(['y_pred', 'probas'], axis=1, inplace=True)


# In[ ]:


print(clfs_results[model_name][2]['best_model'].predict(b_personas.head(1)))
print(clfs_results[model_name][2]['best_model'].predict_proba(b_personas.head(1))[:,1])
print(clfs_results[model_name][2]['best_model'].predict_proba(b_personas.head(1)))


# # PREDICT LISTS OF CLIENTS

# In[ ]:


#Copia de X_test_o para no mancharlo
b_personas = copy.deepcopy(X_test_o)
b_p_negocio = copy.deepcopy(y_test_o)


# In[ ]:


b_personas.columns


# In[ ]:


# revisando que los indices corresponden
print(len(set(X_test_o.index) & set(y_test_o.index)))


# In[ ]:


# revisando el largo de los sets
# print(b_personas.shape, b_p_negocio.shape, probas.shape)

#prediciendo probas y pred para X_test_o
probas = clfs_results[model_name][2]['best_model'].predict_proba(X_test_o)[:,1]
y_pred = clfs_results[model_name][2]['best_model'].predict(X_test_o)

#uniendo predicion y probas a set inicial (faltan datos personales)
b_personas['target'] = b_p_negocio
b_personas['y_pred'] = clfs_results[model_name][2]['best_model'].predict(X_test_o)
b_personas['t_proba'] = clfs_results[model_name][2]['best_model'].predict_proba(X_test_o)[:,1]


#Comprobando las predicciones
results = b_personas


# In[ ]:


bla = personas_original.loc[b_personas.index]
bla


# In[ ]:


set1 = set(b_personas.index)
set2 = set(personas_original.index)

matched = set1.intersection(set2) # set(['dog', 'cat', 'donkey'])
unmatched = set1.symmetric_difference(set2) # set(['pig'])


# In[ ]:


len(matched)


# In[ ]:


print(len(set(personas_original.index) & set(b_personas.index)))


# In[ ]:


new = personas_original.merge(b_personas, left_index=True, right_index=True)
new.shape


# In[ ]:


new.columns.tolist()


# In[ ]:


new2[['t_proba', 'y_pred','target','nro_cot_depto_y','nro_cot_depto_x','is_apellido2_x','is_apellido2_y']].head(20)


# In[ ]:


new2 = new.merge(personas_info, left_on='rut', right_on='rut', how='left')
new2.shape


# In[ ]:


new2.columns.tolist()


# In[ ]:


new2[['rut','nombre','correo','t_proba', 'y_pred','target','nro_cot_depto_y','nro_cot_depto_x']]


# In[ ]:


info = new.merge(personas_info, left_on='rut', right_on='rut', how='left')


# In[ ]:


info[['nombre', 'y_pred', 'target', 't_proba']]


# In[ ]:


personas_original.shape


# In[ ]:


#buscando los datos personales de las personas de X_test_0 con sus indices, se comprueba que estan todos
personas_info.loc[results.index].shape


# In[ ]:


# se traen estos datos desde el conjunto con información personal
new = personas_info.loc[results.index]
# se les agrega las probas y pred hechos con X_test_o
new['t_probas'] = probas
new['y_pred_negocio'] = y_pred
new['target'] = b_personas['target']
new.head()


# In[ ]:


new.columns


# In[ ]:


new[['rut','y_pred_negocio', 't_probas', 'target','rut_original','sexo','celular', 'correo','direccion', 'edad','nombre', 'apellido1', 'apellido2']]


# In[ ]:


#se revisa la inforacion personal
new.head(1)


# # PREDICT FOT PERSON IN LIST

# In[ ]:


# se botan las columnas agregadas al set inicial de información de las personas
new.drop(['y_pred_negocio', 't_probas', 'target'], axis=1, inplace=True)


# In[ ]:


#se toma la primera persona en la lista
to_predict = new.head(1)
#se escogen las variables predictoras
to_predict = to_predict[is_no_time_price]
#se realiza la dummizacion
to_predict = pd.get_dummies(to_predict)
# se vericica el largo de las filas del dummy para que calzen con las del modelo
to_predict.shape


# In[ ]:


# se predice con el modelo para la persona anterior
clfs_results[model_name][2]['best_model'].predict_proba(to_predict)[:,1]


# In[ ]:


personas_info.merge(x, left_index=True, right_index=True).columns.tolist()


# In[ ]:


m = personas_original.merge(X_test_o, left_index=True, right_index=True)
m['t_proba'] = clfs_results[model_name][2]['best_model'].predict_proba(X_test_o)[:,1]
m['y_pred'] = clfs_results[model_name][2]['best_model'].predict(X_test_o)
m[['rut','t_proba', 'y_pred', 'negocio']]


# In[ ]:


def format_rut(x):
    rut = x.split('-')
    l_s = list(rut[0])
    r_l = l_s[::-1]
    new = []
    i = 0
    new.append(rut[1])
    new.append('-')
    for c in r_l:
        if i==3:
            new.append('.')
            i = 0
        new.append(c)
        i +=1
    rut = ''.join(new[::-1])
    return rut

def negocio_rank_callback(modelo, rut):
    tmp_personas = copy.deepcopy(personas_info)

    predictors = no_is_time_price

    rut = format_rut(rut) 
    
    to_predict = tmp_personas[tmp_personas.rut == rut]
    print(to_predict)
    index = to_predict.index
    # target = tmp_personas.negocio
    p_personas = tmp_personas[predictors]
    d_personas = pd.get_dummies(p_personas)

    to_predict = d_personas.loc[index]

    y_proba = modelo.predict_proba(to_predict)
    y_pred = modelo.predict(to_predict)
    print('Negocio', y_proba, y_pred)
#     p = html.H3('%.4f' % y_proba[:,1])
    return p


# In[ ]:


modelo = clfs_results[model_name][2]['best_model']
negocio_rank_callback(modelo, "10033390-2")


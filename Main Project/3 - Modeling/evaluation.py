import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import (plot_lift_curve, plot_gain_curve, plot_classifier_insights,
	cross_roc_validation, plot_confusion_matrix, plot_combo_chart, cross_lift_validation,
	plot_roc_curve, cross_lift50_validation)

from sklearn.externals.joblib import load
from settings import base_settings
from predictors import predictors_set

import plotly
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio

import warnings
warnings.filterwarnings('ignore')

def save_feature_graph(feature):
	print('Ploting {}'.format(feature))
	figure_combo = plot_combo_chart(feature, model, personas, TARGET_PREDICT)
	pio.write_image(figure_combo, '{}//plots//{}_combo_{}.png'.format(path, model_name, feature))
	plotly.offline.plot(figure_combo, filename='{}//plots//{}_combo_{}.html'.format(path, model_name, feature),
			auto_open=False)
	print('Written combo chart feature {}'.format(feature))

if __name__ == '__main__':
	username, api_key = 'jcardenas.lie', 'lr1c37zw81'
	plotly.tools.set_credentials_file(username=username, api_key=api_key)
	pio.orca.config.port = 8999

	# SETTINGS
	RANDOM_STATE, N_JOBS = base_settings['RANDOM_STATE'], base_settings['N_JOBS']
	FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS = base_settings['FOLDER_RESULTS'], \
	base_settings['TARGET_PREDICT'], base_settings['USE_PREDICTORS']
	SCORER = base_settings['SCORER']
	

	#DATA LOAD
	X_grid = pd.read_csv('{}//x_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT, USE_PREDICTORS), index_col=[0])
	y_grid = pd.read_csv('{}//y_grid_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0], header=None)
	X_test_o = pd.read_csv('{}//x_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0])
	y_test_o = pd.read_csv('{}//y_test_{}_{}.csv'.format(FOLDER_RESULTS, TARGET_PREDICT,USE_PREDICTORS), index_col=[0], header=None)

	personas = pd.read_csv('..\\..\\Datos\\experiments\\{}'.format(base_settings['DATASET_FILE_NAME']), index_col=[0], encoding = "ISO-8859-1")

	COL_PREDICTORS = X_grid.columns

	print('X Grid Shape',X_grid.shape)
	print('X Test Shape',  X_test_o.shape)
	print('Y Grid Shape',y_grid.shape )
	print('Y Test Shape', y_test_o.shape)
	print(type(X_grid), type(y_grid))


	print('Using: ', USE_PREDICTORS, TARGET_PREDICT)
	
	# CARGANDO MODELOS #############################################################################################################
	models = {}
	
	## FOLDER RESULTS : "resultados\\isno\\negocio"
	path = "{}//Models//{}".format(FOLDER_RESULTS, SCORER)
	
	for name in os.listdir(path):
		if name.endswith(".joblib"):
			# print('{}/{}'.format(path, name))
			print('Model Loaded {}'.format(name))
			model_name = "".join(list(name)[:2])
			models[model_name] = load("{}/{}".format(path, name))
	
	
	#PLOTS FOLDER CREATION ########################################################################################################	
	for model_name in models.keys():
		if not os.path.exists('{}//plots'.format(path)):
			os.mkdir('{}//plots'.format(path))
			print('CREATED {}//plots'.format(path))

	for model_name in models.keys():
		if not os.path.exists('{}//plots//{}'.format(path, model_name)):
			os.mkdir('{}//plots//{}'.format(path, model_name))
			print('CREATED {}//plots//{}'.format(path, model_name))
	

	###################################################################################################################
	## ROC CURVES VALIDACION CRUZADA ######################################################################################################################
	# mean_rocs = dict()
	# results_list = []
	# results_dict = {}
	# for model_name in models.keys():
	#     folder_plots = '{}//plots//{}'.format(path, model_name)
	#     print('ROC Evaluation ', model_name)
	#     model = models[model_name]
	#     # CROSS VALIDATION
	#     mean_fpr, mean_tpr, results = cross_roc_validation(model, model_name, folder_plots, 
	#                                                        X_grid, y_grid, X_test_o, y_test_o, splits=6)
	#     mean_rocs[model_name] = dict()
	#     mean_rocs[model_name]['fpr'] = mean_fpr
	#     mean_rocs[model_name]['tpr'] = mean_tpr
	#     print(results)
	#     results_list.append(results)
	#     results_dict[model_name] = results


	# cv_results = pd.DataFrame(results_list)
	# cv_results.to_excel('{}//cv_results.xlsx'.format(path))
	
	## ALL ROC CURVES ######################################################################################################################
	# plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
 #             label='Chance', alpha=.8)
	# for key in mean_rocs.keys():
	#     fpr = mean_rocs[key]['fpr']
	#     tpr = mean_rocs[key]['tpr']
	#     plt.plot(fpr, tpr, lw=1, label=key + ' AUC ' + results_dict[key]['auc'])

	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('All models ROC Curve')
	# plt.legend(loc="lower right",  bbox_to_anchor=(1, 0))
	# plt.savefig('{}//plots//all_models_roc.png'.format(path), bbox_inches = 'tight', dpi=300)
	# plt.close()

	# LIFT50 CURVES ###################################################################################################
	mean_gain = dict()
	results_list = []
	results_dict = {}
	for model_name in models.keys():
	    folder_plots = '{}//plots//{}'.format(path, model_name)
	    print('Lift50 Evaluation ', model_name)
	    model = models[model_name]
	    # CROSS LIFT VALIDATION
	    gain, percentages = cross_lift50_validation(model, model_name, folder_plots, 
	                                 X_grid, y_grid, splits=6)
	    mean_gain[model_name] = dict()
	    mean_gain[model_name]['gain'] = gain
	    mean_gain[model_name]['percentages'] = percentages


	plt.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')
	for key in mean_gain.keys():
	    gains2 = mean_gain[key]['gain']
	    
	    plt.plot(percentages, gains2, lw=3, alpha=0.8, label=key)

	plt.ylabel('Lift')
	plt.xlabel('% Población')
	plt.title('All models Lift Curve')
	plt.legend(loc="lower right",  bbox_to_anchor=(1, 0))
	plt.savefig('{}//plots//all_models_lift50.png'.format(path), bbox_inches = 'tight', dpi=300)
	plt.close()

	## LIFT CURVES ###################################################################################################
	# mean_gain = dict()
	# results_list = []
	# results_dict = {}
	# for model_name in models.keys():
	#     folder_plots = '{}//plots//{}'.format(path, model_name)
	#     print('Lift Evaluation ', model_name)
	#     model = models[model_name]
	#     # CROSS LIFT VALIDATION
	#     gain, percentages = cross_lift_validation(model, model_name, folder_plots, 
	#                                  X_grid, y_grid, X_test_o, y_test_o, splits=6)
	#     mean_gain[model_name] = dict()
	#     mean_gain[model_name]['gain'] = gain
	#     mean_gain[model_name]['percentages'] = percentages


	# plt.plot([0, 1], [1, 1], 'k--', lw=2, label='Baseline')
	# for key in mean_gain.keys():
	#     gains2 = mean_gain[key]['gain']
	    
	#     plt.plot(percentages, gains2, lw=3, alpha=0.8, label=key)

	# plt.ylabel('Lift')
	# plt.xlabel('% Población')
	# plt.title('All models Lift Curve')
	# plt.legend(loc="lower right",  bbox_to_anchor=(1, 0))
	# plt.savefig('{}//plots//all_models_lift.png'.format(path), bbox_inches = 'tight', dpi=300)
	# plt.close()




	# GAIN, LIFT, CONF, NORM CONF, INSIGHTS: GAIN, LIFT, PRE-REC, FEATURE-IMPORTANCE ####################################################################################################################
	# for model_name in models.keys():
	# 	print('Plotting {}'.format(model_name))
	# 	folder_plots = '{}//plots//{}'.format(path, model_name)
	# 	model = models[model_name]
	# 	# MODEL INSIGHST GAIN LIFT FEATURE-IMPORTANCE PRE-REC KS-STATISTICS
	# 	plot_classifier_insights(X_test_o, y_test_o, model, COL_PREDICTORS,
	# 		folder=folder_plots, 
	# 		model_name=model_name)

	# 	# CONF MATRIX
	# 	classes = base_settings['CLASSES']
	# 	y_probas = model.fit(X_grid, y_grid).predict_proba(X_test_o)
	# 	y_pred=model.predict(X_test_o)

	# 	plot_confusion_matrix(y_test_o, y_pred, classes, folder_plots, model_name, normalize=False)
	# 	plot_confusion_matrix(y_test_o, y_pred, classes, folder_plots, model_name, normalize=True)
	# 	title = '{} Test ROC Curve'.format(model_name)
		# plot_roc_curve(model, X_test_o, y_test_o, folder=folder_plots, title=title)
	# 	# # GAIN PLOT
	# 	figure_lift = plot_gain_curve(model_name, model, X_test_o, y_test_o)
	# 	pio.write_image(figure_lift, '{}//plots//{}_gain_curve.png'.format(path, model_name))

	#COMBO CHART FEATURE IMPORTANCE #######################################################################
	# pick best clasifier

	# if USE_PREDICTORS == 'isno' and TARGET_PREDICT == 'negocio':
	# 	model = models['XG']; print('model XGB')
	# 	model_name = 'XG'
	# elif USE_PREDICTORS == 'isno' and TARGET_PREDICT == 'compra':
	# 	model = models['RF']; print('model RF')
	# 	model_name = 'RF'
	# elif USE_PREDICTORS == 'nois' and TARGET_PREDICT == 'negocio':
	# 	model = models['RF']; print('model RF')
	# 	model_name = 'RF'
	# elif USE_PREDICTORS == 'nois' and TARGET_PREDICT == 'compra':
	# 	model = models['RF']; print('model RF')
	# 	model_name = 'RF'
	
	# mask = (personas[base_settings['PROYECT_SELECT']] > 0)
	# personas = personas[mask]
	
	# try:
	# 	feature_importances = pd.DataFrame(model.feature_importances_,
 #                                   index = X_grid.columns,
 #                                    columns=['importance']).sort_values('importance', ascending=False)
	# 	print('Top 5 feature importance :', feature_importances.index[:4])

	# 	save_feature_graph(feature_importances.index[0])
	# 	save_feature_graph(feature_importances.index[1])
	# 	save_feature_graph(feature_importances.index[2])
		
	# except:
	# 	print('ERROR: Feature importance not supported')
		


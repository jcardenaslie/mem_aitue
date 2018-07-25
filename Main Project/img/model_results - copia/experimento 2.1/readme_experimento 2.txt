{'readme': {
	'experiment_name': 'experimento 2', 
	'experiment_dataset': 'personas_cotizacion3.csv', 
	'personas_shape': (2251, 34), 
	'personas_columns': ['Unnamed: 0', 'is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac', 'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_profesion', 'is_telefono', 'loc_comuna', 'loc_provincia', 'loc_region', 'max_rango_edad', 'mean_cot_bod', 'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 'negocio', 'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu', 'nro_proyectos', 'profesion', 'rut', 'sexo', 'tipo_cliente', 'Altos del Valle', 'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610', 'Edificio Junge'], 'personas_col_drop': ['Unnamed: 0', 'rut', 'negocio', 'max_rango_edad', 'is_profesion'], 'personas_after_drop_columns': Index(['is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac',
       'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_telefono',
       'loc_comuna', 'loc_provincia', 'loc_region', 'mean_cot_bod',
       'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 'nro_cot_bod',
       'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu', 'nro_proyectos',
       'profesion', 'sexo', 'tipo_cliente', 'Altos del Valle',
       'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610',
       'Edificio Junge'],
      dtype='object'), 
	'personas_predictors': Index(['is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac',
       'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_telefono',
       'loc_comuna', 'loc_provincia', 'loc_region', 'mean_cot_bod',
       'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 'nro_cot_bod',
       'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu', 'nro_proyectos',
       'profesion', 'sexo', 'tipo_cliente', 'Altos del Valle',
       'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610',
       'Edificio Junge'],
      dtype='object'), 
	'personas_dummies_shape': (2251, 172), 
	'personas_negocio': (714,), 
	'personas_nonegocio': (1537,), 
	'x_train_shape': (1575, 172), 
	'y_train_shape': (1575,), 
	'x_test_shape': (676, 172), 
	'y_test_shape': (676,), 
	'model': {
		'PRT_LR': {
			'conf_matrix': array([[387,  80],
			   [ 38, 171]], dtype=int64), 
			'cla_report': '             precision    recall  f1-score   support\n\n      False       0.91      0.83      0.87       467\n       True       0.68      0.82      0.74       209\n\navg / total       0.84      0.83      0.83       676\n'}, 'KNN': {'name': 'KNN', 'best_estimator': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
				   metric_params=None, n_jobs=1, n_neighbors=9, p=2,
				   weights='uniform'), 
			'best_score': 0.8089737894269213, 
			'conf_matrix': array([[420,  47],
			   [ 84, 125]], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.83      0.90      0.87       467\n       
				True       0.73      0.60      0.66       209\n\navg / 
				total       0.80      0.81      0.80       676\n'}, 
		'PCAKNN': {
			'name': 'PCAKNN', 
			'best_estimator': LogisticRegression(C=0.4393970560760795, class_weight=None, dual=False,
				  fit_intercept=True, intercept_scaling=1, max_iter=100,
				  multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
				  solver='liblinear', tol=0.0001, verbose=0, warm_start=False), 
			'best_score': 0.8334073745002222, 
			'conf_matrix': array([[423,  44],
			   [ 72, 137]], dtype=int64), 
			   'cla_report': '             
						precision    recall  f1-score   support\n\n      
			   False       0.85      0.91      0.88       467\n       
			   True       0.76      0.66      0.70       209\n\navg / 
			   total       0.82      0.83      0.82       676\n'}, 
		'ULR': {
			'conf_matrix': array([[417,  50],
			   [ 69, 140]], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.86      0.89      0.88       467\n       
				True       0.74      0.67      0.70       209\n\navg / 
				total       0.82      0.82      0.82       676\n'}, 
		'BLR': {
			'conf_matrix': array([
				[382,  85],
			   [ 49, 160]
			], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.89      0.82      0.85       467\n       
				True       0.65      0.77      0.70       209\n\navg / 
				total       0.81      0.80      0.81       676\n'}, 
			'DT': {
				'name': 'DT', 
				'best_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=11,
						max_features=29, max_leaf_nodes=None,
						min_impurity_decrease=0.0, min_impurity_split=None,
						min_samples_leaf=9, min_samples_split=2,
						min_weight_fraction_leaf=0.0, presort=False, random_state=None,
						splitter='best'), 
				'best_score': 0.8378498445135495, 
				'conf_matrix': array([
					[417,  50],
				   [73, 136]
				], dtype=int64), 
				'cla_report': '             
							precision    recall  f1-score   support\n\n      
					False       0.85      0.89      0.87       467\n       
					True       0.73      0.65      0.69       209\n\navg / 
					total       0.81      0.82      0.81       676\n'}, 
		'URF': {
			'name': 'URF', 
			'best_estimator': RandomForestClassifier(bootstrap=False, class_weight=None,
				criterion='entropy', max_depth=22, max_features='sqrt',
				max_leaf_nodes=None, min_impurity_decrease=0.0,
				min_impurity_split=None, min_samples_leaf=1,
				min_samples_split=24, min_weight_fraction_leaf=0.0,
				n_estimators=90, n_jobs=1, oob_score=False, random_state=None,
				verbose=0, warm_start=False), 
			'best_score': 0.8462905375388716, 
			'conf_matrix': array([
				[440,  27],
			   [ 80, 129]
			], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.85      0.94      0.89       467\n       
				True       0.83      0.62      0.71       209\n\navg / 
				total       0.84      0.84      0.83       676\n'}, 
			'BRF': {
			'conf_matrix': array([
				[382,  85],
				[ 49, 160]
			], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.89      0.82      0.85       467\n       
				True       0.65      0.77      0.70       209\n\navg / 
				total       0.81      0.80      0.81       676\n'}, 
		'SVM': {
			'name': 'SVM', 
			'best_estimator': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
				decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
				max_iter=-1, probability=False, random_state=None, shrinking=True,
				tol=0.001, verbose=False), 
			'best_score': 0.8214127054642382, 
			'conf_matrix': array([
				[410,  57],
			   [ 74, 135]
			], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
				False       0.85      0.88      0.86       467\n       
				True       0.70      0.65      0.67       209\n\navg / 
				total       0.80      0.81      0.80       676\n'}}, 
		'time_exec': 387.2842290401459}}
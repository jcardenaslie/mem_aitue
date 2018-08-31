{'readme': {
	'experiment_name': 'experimento 3', 
	'experiment_dataset': 'personas_cotizacion4.csv', 
	'personas_shape': (6184, 34), 
	'personas_columns': [
		'Unnamed: 0', 'is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac', 'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_profesion', 
		'is_telefono', 'loc_comuna', 'loc_provincia', 'loc_region', 'max_rango_edad', 'mean_cot_bod', 'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu', 
		'negocio', 'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu', 'nro_proyectos', 'profesion', 'rut', 'sexo', 'tipo_cliente', 'Altos del Valle', 
		'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610', 'Edificio Junge'], 'personas_col_drop': ['Unnamed: 0', 'rut', 'negocio', 'max_rango_edad'], 
	'personas_after_drop_columns': Index(['is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac',
		'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_profesion',
		'is_telefono', 'loc_comuna', 'loc_provincia', 'loc_region',
		'mean_cot_bod', 'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu',
		'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu',
		'nro_proyectos', 'profesion', 'sexo', 'tipo_cliente', 'Altos del Valle',
		'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610',
		'Edificio Junge'],
		dtype='object'), 
	'personas_predictors': Index(['is_apellido1', 'is_appelido2', 'is_celular', 'is_direccion', 'is_fnac',
		'is_nombre', 'is_nombrecompleto', 'is_nrofam', 'is_profesion',
		'is_telefono', 'loc_comuna', 'loc_provincia', 'loc_region',
		'mean_cot_bod', 'mean_cot_depto', 'mean_cot_esta', 'mean_cot_estu',
		'nro_cot_bod', 'nro_cot_depto', 'nro_cot_esta', 'nro_cot_estu',
		'nro_proyectos', 'profesion', 'sexo', 'tipo_cliente', 'Altos del Valle',
		'Edificio Urban 1470', 'San Andres Del Valle', 'Edificio Mil610',
		'Edificio Junge'],
		dtype='object'), 
	'personas_dummies_shape': (6184, 174), 
	'personas_negocio': (1101,), 
	'personas_nonegocio': (5083,), 
	'x_train_shape': (4328, 174), 
	'y_train_shape': (4328,), 
	'x_test_shape': (1856, 174), 
	'y_test_shape': (1856,), 
	'model': {
		PRT_LR': {
			'conf_matrix': array([[1328,  193],
			[  77,  258]], dtype=int64), 
			'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.95      0.87      0.91      1521\n       
			True       0.57      0.77      0.66       335\n\navg / 
			total       0.88      0.85      0.86      1856\n'}, 
	   'KNN': {
		   'name': 'KNN', 
		   'best_estimator': KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
			   metric_params=None, n_jobs=1, n_neighbors=9, p=2,
			   weights='uniform'), 
			   'best_score': 0.8732212160413971, 
			   'conf_matrix': array([[1469,   52],
			[ 171,  164]], dtype=int64), 
			'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.90      0.97      0.93      1521\n       
			True       0.76      0.49      0.60       335\n\navg / 
			total       0.87      0.88      0.87      1856\n'}, 
	   'PCAKNN': {
			'name': 'PCAKNN', 
			'best_estimator': LogisticRegression(C=0.4393970560760795, class_weight=None, dual=False,
			  fit_intercept=True, intercept_scaling=1, max_iter=100,
			  multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
			  solver='liblinear', tol=0.0001, verbose=0, warm_start=False), 
			  'best_score': 0.8921410090556274, 
			  'conf_matrix': array([[1462,   59],
			[ 155,  180]], dtype=int64), 'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.90      0.96      0.93      1521\n       
			True       0.75      0.54      0.63       335\n\navg / 
			total       0.88      0.88      0.88      1856\n'}, 
	   'ULR': {
			'conf_matrix': array([[1461,   60],
			[ 150,  185]], dtype=int64), 'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.91      0.96      0.93      1521\n       
			True       0.76      0.55      0.64       335\n\navg / 
			total       0.88      0.89      0.88      1856\n'}, 
	   'BLR': {
			'conf_matrix': array([[1335,  186],
			[  81,  254]], dtype=int64), 'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.94      0.88      0.91      1521\n       
			True       0.58      0.76      0.66       335\n\navg / 
			total       0.88      0.86      0.86      1856\n'}, 
	   'DT': {
		   'name': 'DT', 
		   'best_estimator': DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
				max_features=29, max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None,
				min_samples_leaf=6, min_samples_split=2,
				min_weight_fraction_leaf=0.0, presort=False, random_state=None,
				splitter='best'), 
				'best_score': 0.8876131953428201, 
				'conf_matrix': array([[1472,   49],
		   [ 200,  135]], dtype=int64), 
		   'cla_report': '             
					precision    recall  f1-score   support\n\n      
			False       0.88      0.97      0.92      1521\n       
			True       0.73      0.40      0.52       335\n\navg / 
			total       0.85      0.87      0.85      1856\n'}, 
	   'URF': {
		   'name': 'URF', 
		   'best_estimator': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
				max_depth=26, max_features='auto', max_leaf_nodes=None,
				min_impurity_decrease=0.0, min_impurity_split=None,
				min_samples_leaf=1, min_samples_split=15,
				min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=1,
				oob_score=False, random_state=None, verbose=0,
				warm_start=False), 
				'best_score': 0.8935963777490298, 
				'conf_matrix': array([[1481,   40],
		   [ 158,  177]], dtype=int64), 
		   'cla_report': '             
					precision    recall  f1-score   support\n\n      
		   False       0.90      0.97      0.94      1521\n       
		   True       0.82      0.53      0.64       335\n\navg / 
		   total       0.89      0.89      0.88      1856\n'}, 
	   'BRF': {
		   'conf_matrix': array([[1335,  186],
		   [  81,  254]], dtype=int64), 
		   'cla_report': '             
					precision    recall  f1-score   support\n\n      
		   False       0.94      0.88      0.91      1521\n       
		   True       0.58      0.76      0.66       335\n\navg / 
		   total       0.88      0.86      0.86      1856\n'}, 
	   'SVM': {
			'name': 'SVM', 
			'best_estimator': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
				decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
				max_iter=-1, probability=False, random_state=None, shrinking=True,
				tol=0.001, verbose=False), 
			'best_score': 0.8826002587322122, 
			'conf_matrix': array([[1456,   65],
			[ 155,  180]], dtype=int64), 
			'cla_report': '             
						precision    recall  f1-score   support\n\n      
			   False       0.90      0.96      0.93      1521\n       
			   True       0.73      0.54      0.62       335\n\navg / 
			   total       0.87      0.88      0.87      1856\n'}}, 
	   'time_exec': 1150.467589378357}}
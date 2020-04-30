class ModelRunningClass:
    def model_running_random_forest_regressor_gridsearch_sklearn():
		# Create the parameter for GridSearch
		param_grid = {
    	'bootstrap': [True],
    	'max_depth': [1, 2, 3, 4],
    	'max_features': [2, 3]
		}
		# Create a based model
		rf = RandomForestRegressor()
		# Instantiate the grid search model
		grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
		grid_search.fit(X, y)
    return grid_search

    def model_running_random_forest_regressor_gridsearch_Dask():
    	#starting cluster
    	from dask.distributed import Client, progress
		client = Client(processes=False, threads_per_worker=2, n_workers=1, memory_limit='4GB')
		#Dask GridSearch running
		with joblib.parallel_backend('dask'):
    		grid_search.fit(X, y)
    return grid_search
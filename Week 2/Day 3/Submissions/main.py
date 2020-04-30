from data_loader import DataLoaderClass
from data_cleaning import DataCleaningClass
from model_running import ModelRunningClass
import utils

def main():
    print("Setting up data directory")
    print("-------------------------")
    #Data loading
    DataLoaderClass.load_flights_data()
    #Data cleaning
    DataCleaningClass.cleaning_flights_data()
    #Model running
    ModelRunningClass.model_running_random_forest_regressor_gridsearch_sklearn()
    #Dask GridSearch running
    ModelRunningClass.model_running_random_forest_regressor_gridsearch_Dask()
    print('Done! Impovement will coming...')


if __name__ == '__main__':
    main()

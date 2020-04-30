class DataCleaningClass:
    def cleaning_flights_data():
    	#loading to dataframe
    	data = pd.read_csv('data/nycflights/1990.csv')
    	data = data.fillna(0)
    	data = data.select_dtypes(exclude=['object'])
    	X = data.drop(['DepDelay'], axis=1)
		y = data.DepDelay

    return X,y
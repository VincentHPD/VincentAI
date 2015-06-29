"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
import pandas as pd
import json, sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import f1_score
from sklearn.learning_curve import learning_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import vincent
start_time = time.time()

crime_database_file = 'crime_records.db'

type_mapper = vincent.Mapper()
beat_mapper = vincent.Mapper()

#create and add crime instances to the database
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT, NOffenses INTEGER FROM HPDCrimes ORDER BY Year ASC')
crime_dicts = {}
for crime in db_cur.fetchall():
	(year, month, m_day, w_day) = (crime[0], crime[1], crime[2], crime[3])
	#convert the OffenseTypes and Beats to hashes
	beat_hash = beat_mapper.get_hash(crime[4])
	type_hash = type_mapper.get_hash(crime[5])

	n_offenses = crime[6] #the number of offenses which occured (target value)

	temp_dict = {
	    'date' : pd.datetime(year, month, m_day),
	    'beat_hash' : beat_hash,
	    'type_hash' : type_hash,
	    'n_offenses' : n_offenses}

	crime_key = '{}-{}-{}-{}-{}'.format(year, month, m_day, beat_hash, type_hash)
	crime_dicts.update({crime_key : temp_dict})
db_con.close()

#find the earliest and latest datetime objects
all_crime_dates = [crime_dicts[key]['date'] for key in crime_dicts]
sorted_crime_dates = sorted(all_crime_dates)
#find all of the hashes for beats and types of crimes
beat_mapper_hashes = beat_mapper.hash_to_key.keys()
type_mapper_hashes = type_mapper.hash_to_key.keys()

print("Types of crimes: {}".format(type_mapper.key_to_hash))

#create an array to store all possible incidences of crime whether they occurred or not
start_date, end_date = sorted_crime_dates[0], sorted_crime_dates[-1]
range_dates = pd.date_range(start_date, end_date, freq='D')

no_crime_dicts = {} #stores entries for combinations where no crimes occured
#find all combinations where a crime didn't occur
for date_ in range_dates:
	for beat in beat_mapper_hashes:
		for crime_type in type_mapper_hashes:
			temp_key = '{}-{}-{}-{}-{}'.format(date_.year, date_.month, date_.day, beat, crime_type)
			temp_dict = {temp_key : {'date' : date_, 'beat_hash' : beat, 'type_hash' : crime_type, 'n_offenses' : 0.0 } }
			#check to see if this combination already has a crime associated
			if not crime_dicts.has_key(temp_key):
				#if not, add it as a no crime occurrence (n_offenses = 0.0)
				no_crime_dicts.update(temp_dict)
#combine the data of all crime and no-crime occurences
major_data_dict = crime_dicts.copy()
major_data_dict.update(no_crime_dicts)

#DEBUG
"""for key in major_data_dict.keys()[:50]:
	print('{} {} {} {}'.format(major_data_dict[key]['date'], beat_mapper.get_key(major_data_dict[key]['beat_hash']),
		type_mapper.get_key(major_data_dict[key]['type_hash']), major_data_dict[key]['n_offenses']))
"""

#make a nested list containing all of the data
xy_list = []
for key in major_data_dict:
	#exclude year and extract day of week
	month, m_day, w_day = major_data_dict[key]['date'].month, major_data_dict[key]['date'].day, major_data_dict[key]['date'].weekday()
	beat_hash = major_data_dict[key]['beat_hash']
	type_hash = major_data_dict[key]['type_hash']
	n_offenses = major_data_dict[key]['n_offenses']
	xy_list.append( [month, m_day, w_day, beat_hash, type_hash, n_offenses] )

#make the numpy array of the data
xy_array = np.array(xy_list)

#seperate the features from the target to make X and y
split_xy_array = np.hsplit(xy_array, len(xy_array[0]))
X_data = np.hstack( (split_xy_array[0], split_xy_array[1], split_xy_array[2], split_xy_array[3], split_xy_array[4]) )
y_data = np.ravel(split_xy_array[5])

#create feature scaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_data_scaled = scaler.fit_transform(X_data)

#create regressor
#regr = LinearRegression()
#regr = SGDClassifier(penalty='l2', alpha=0.00003, eta0=0.01)
regr = KNeighborsRegressor(n_neighbors = 10, algorithm = 'auto')

#cross validation
accuracy_rates = []
kf = KFold(len(y_data), n_folds = 2, shuffle=True) #create cross validation model
"""
for train_index, test_index in kf:
    X_train, X_test = X_data_scaled[train_index], X_data_scaled[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    regr.fit(X_train, y_train)
    predicted = regr.predict(X_test)
    accuracy = regr.score(X_test, y_test)
    accuracy_rates.append(accuracy)

print("Mean(accuracy_rates) = %.5f" % (np.mean(accuracy_rates)))
"""
#split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data_scaled, y_data, test_size = 0.4, random_state=42)

#fit the regressor
regr.fit(X_train, y_train)

#print('f1 scores: {}'.format(f1_score(y_data, regr.predict(X_data_scaled), average=None)))


#plotting learning curves
plt.figure()
plt.title('Learning Curves for: {}'.format([t for t in type_mapper.key_to_hash]))
plt.xlabel('Training examples')
plt.ylabel('Score')
train_sizes, train_scores, test_scores = learning_curve(regr, X_data_scaled, y_data, train_sizes=np.array([.1,.2,.5,.8,.99]), cv=kf)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()


"""
#plotting
pca = PCA(n_components=1)
reduc_x = pca.fit_transform(X_data_scaled)
reduc_x = np.ravel(reduc_x)
plt.figure(1)
plt.subplot(1, 1, 1)
print '{} {}'.format(reduc_x, y_data)
plt.plot(reduc_x, y_data, 'ro')
plt.ylim([-.1, 20.1])
plt.xlim([-3, 3])
plt.title('Vincent')
plt.xlabel('Reduced Feature Space')
plt.ylabel('Number of Offenses')
plt.show()
"""

print 'time to complete: %ds' % (time.time() - start_time)

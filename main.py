"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
import pandas as pd
import json, sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import vincent
start_time = time.time()

crime_database_file = 'crime_records.db'

type_mapper = vincent.Mapper()
beat_mapper = vincent.Mapper()

#create and add crime instances to the database
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT FROM HPDCrimes ORDER BY Year ASC')
crime_dicts = {} 
for crime in db_cur.fetchall():
    (year, month, m_day, w_day) = (crime[0], crime[1], crime[2], crime[3])
    #convert the OffenseTypes and Beats to hashes
    beat_hash = beat_mapper.get_hash(crime[4])
    type_hash = type_mapper.get_hash(crime[5])
    chance = 1.0 #since we have a record of this crime we know the probability was 1.0 (target value) 
    temp_dict = { 
	    'date' : pd.datetime(year, month, m_day),
	    'beat_hash' : beat_hash,
	    'type_hash' : type_hash,
	    'chance' : chance}
    crime_key = '{}-{}-{}-{}-{}'.format(year, month, m_day, beat_hash, type_hash)
    crime_dicts.update({crime_key : temp_dict})
db_con.close()

#find the earliest and latest datetime objects
all_crime_dates = [crime_dicts[key]['date'] for key in crime_dicts]
sorted_crime_dates = sorted(all_crime_dates)
#find all of the hashes for beats and types of crimes
beat_mapper_hashes = beat_mapper.hash_to_key.keys()
type_mapper_hashes = type_mapper.hash_to_key.keys()

#create an array to store all possible incidences of crime whether they occurred or not
start_date, end_date = sorted_crime_dates[0], sorted_crime_dates[-1] 
range_dates = pd.date_range(start_date, end_date, freq='D')

no_crime_dicts = {} #stores entries for combinations where no crimes occured
#find all combinations where a crime didn't occur
for date_ in range_dates:
	for beat in beat_mapper_hashes:
		for crime_type in type_mapper_hashes:
			temp_key = '{}-{}-{}-{}-{}'.format(date_.year, date_.month, date_.day, beat, crime_type)
			temp_dict = {temp_key : {'date' : date_, 'beat_hash' : beat, 'type_hash' : crime_type, 'chance' : 1.0 } } 
			#check to see if this combination already has a crime associated
			if not crime_dicts.has_key(temp_key):
				#if not, update it to reflect no crime occured in that combination and add it as a no crime occurrence
				temp_dict[temp_key]['chance'] = 0.0
				no_crime_dicts.update(temp_dict)
#combine the data of all crime and no-crime occurences
major_data_dict = crime_dicts.copy()
major_data_dict.update(no_crime_dicts)

#DEBUG
"""for key in major_data_dict.keys()[:50]:
	print('{} {} {} {}'.format(major_data_dict[key]['date'], beat_mapper.get_key(major_data_dict[key]['beat_hash']),
		type_mapper.get_key(major_data_dict[key]['type_hash']), major_data_dict[key]['chance']))
"""

"""
split_major_array = np.hsplit(major_array, len(major_array[0]))

#excluding year from dataset
X_data = np.hstack((split_major_array[1], split_major_array[2], split_major_array[3], split_major_array[4]))
y_data = np.ravel(split_major_array[5])

#create feature scaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_data_scaled = scaler.fit_transform(X_data)

accuracy_rates = []
kf = KFold(len(y_data), n_folds = 3, shuffle=False) #create cross validation model

for train_index, test_index in kf:
    X_train, X_test = X_data_scaled[train_index], X_data_scaled[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    accuracy_rates.append(accuracy)

clf.fit(X_data_scaled[:int(len(y_data)*.7)], y_data[:int(len(y_data)*.7)])

print("Mean(accuracy_rates) = %.5f" % (np.mean(accuracy_rates)))

#find and print the f1 score
fsco = f1_score(y_data, clf.predict(X_data_scaled), pos_label = None, average = None) #returns an f1 score for each class respectively, doesn't compute a mean f1score
print("fscores for respective classes are: ")
for i in type_mapper.hash_to_key:
    print("{0}: {1}".format( type_mapper.get_key(i), fsco[i]) )

#predictions for the next week

#find the current epoch time
times = [] #stores the normalized time in seconds of each day for the following days
for i in range(7):
    times.append(time.time() + i*86400)

"""

print 'time to complete: %ds' % (time.time() - start_time)

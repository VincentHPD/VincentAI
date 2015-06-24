"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
import pandas as pd
import json, sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
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

all_data_from_sql = []

for crime in db_cur.fetchall():
    temp_list = []
    temp_list.extend([crime[0], crime[1], crime[2], crime[3]])
    #DEBUG print '{} {} {} {} {} {}'.format(crime[0], crime[1], crime[2], crime[3], crime[4], crime[5])
    #convert the OffenseTypes and Beats to hashes, use a scaling coefficient
    temp_list.append(beat_mapper.get_hash(crime[4]))
    temp_list.append(type_mapper.get_hash(crime[5]))
    #append this temporary list to the main list which stores all the data
    all_data_from_sql.append(temp_list)
db_con.close()

#convert the nested lists into a numpy array
all_data_from_sql = np.asarray(all_data_from_sql)

###find earliest and latest date, and hashes in mapper objects###
earliest_year = all_data_from_sql[:, 0].min()
#find the earliest month (1-12)
earliest_month = 1000 #setting a high default value intentionally
for row in all_data_from_sql[:, :2]:
    if (row[0] == earliest_year) and (row[1] < earliest_month):
        earliest_month = row[1]
#find the earliest day (1-31) within the earliest month within the earliest year
earliest_day = 50 #setting a high default value intentionally
for row in all_data_from_sql[:, :3]:
    if (row[0] == earliest_year) and (row[1] == earliest_month) and (row[2] < earliest_day):
        earliest_day = row[2]

latest_year = all_data_from_sql[:, 0].max()
latest_month = 0 #setting a low default value intentionally
#find the latest month (1-12) within the latest year
for row in all_data_from_sql[:, :2]:
    if (row[0] == latest_year) and (row[1] > latest_month):
        latest_month = row[1]
#find the latest day (1-31) within the latest month within the latest year
latest_day = 0 #setting a low default value intentionally
for row in all_data_from_sql[:, :3]:
    if (row[0] == latest_year) and (row[1] == latest_month) and (row[2] > latest_day):
        latest_day = row[2]

beat_mapper_hashes = beat_mapper.hash_to_key.keys()
type_mapper_hashes = type_mapper.hash_to_key.keys()

print("{} {} {}".format(earliest_year, earliest_month, earliest_day))
print("{} {} {}".format(latest_year, latest_month, latest_day))
print("{} {}".format(beat_mapper_hashes, type_mapper_hashes))

###_###

#create an array to store all possible incidences of crime whther they occurred or not

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


print 'time to complete: %ds' % (time.time() - start_time)
"""

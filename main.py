"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
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

major_array = np.vstack(all_data_from_sql)
split_major_array = np.hsplit(major_array, 6)

X_data = np.hstack((split_major_array[0], split_major_array[1], split_major_array[2], split_major_array[3], split_major_array[4]))
y_data = np.ravel(split_major_array[5])

#create feature scaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_data_scaled = scaler.fit_transform(X_data)

accuracy_rates = []
kf = KFold(len(y_data), n_folds = 3, shuffle=False) #create cross validation model
clf = SGDClassifier(alpha=0.000001, loss='log', fit_intercept = False, shuffle=False)
#clf = KNeighborsClassifier(10)

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

#create a dictionary to store future crimes
fut_week_crimes = dict()

fut_week_crimes_list = []
for time_i in times:
    year = time.localtime(time_i).tm_year
    month = time.localtime(time_i).tm_mon
    mday = time.localtime(time_i).tm_mday
    wday = time.localtime(time_i).tm_wday
    for beat_id in beat_mapper.hash_to_key:
        #reverse hash lookup to display strings for beats and types of crimes
        temp_dict = {
            'date' : time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_i)),
            'beat': beat_mapper.get_key(beat_id),
            'type': type_mapper.get_key(clf.predict([year, month, mday, wday, beat_id])[0])
            }
        fut_week_crimes_list.append(temp_dict)
fut_week_crimes["crimes"] = fut_week_crimes_list
with open('future.json', 'w') as fl:
    json.dump(fut_week_crimes, fl)

print 'time to complete: %ds' % (time.time() - start_time)

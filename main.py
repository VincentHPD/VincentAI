"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
import json, sqlite3
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import vincent
start_time = time.time()

crime_database_file = 'crime_records.db'

type_mapper = vincent.Mapper()
beat_mapper = vincent.Mapper()

#create and add crime instances to the database
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT eTime, OffenseType, Beat, NumOffenses FROM HPDCrimes ORDER BY eTime ASC')

all_data_from_sql = []

base_time = 0.0 #this is base line from where time will be referenced
for ctr, crime in enumerate(db_cur.fetchall()):
    temp_list = []
    #find the base time
    if ctr == 0:
        base_time = float(crime[0])
    #add the time with respect to base_time
    temp_list.append(crime[0] - base_time)
    #convert the OffenseTypes and Beats to hashes, use a scaling coefficient
    temp_list.append(type_mapper.get_hash(crime[1]))
    temp_list.append(beat_mapper.get_hash(crime[2]))
    #append this temporary list to the main list which stores all the data
    #print('{} {} {}'.format(crime[0], crime[1], crime[2]))
    all_data_from_sql.append(temp_list)

major_array = np.vstack(all_data_from_sql)
split_major_array = np.hsplit(major_array, 3)

X_data = np.hstack((split_major_array[0], split_major_array[2]))
y_data = np.ravel(split_major_array[1])

#normalize data
#X_data = preprocessing.scale(X_data)
#y_data = preprocessing.scale(y_data)

accuracy_rates = []
kf = KFold(len(y_data), n_folds = 3, shuffle=False) #create cross validation model
clf = KNeighborsClassifier(5)
ctr = 0
for train_index, test_index in kf:
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    accuracy_rates.append(accuracy)

    if ctr == -1:
        plt.plot(X_test.transpose()[0], predicted, 'b')
        plt.plot(X_test.transpose()[0], y_test, 'r')
        plt.axis([0, 1.5e9, 0, 7])
        plt.show()
        ctr += 1

clf.fit(X_data[:int(len(y_data)*.7)], y_data[:int(len(y_data)*.7)])

print("Mean(accuracy_rates) = %.5f" % (np.mean(accuracy_rates)))

#account for base time
#reverse hash lookup

#predictions for the next week

#find the current epoch time
times = [] #stores the normalized time in seconds of each day for the following days
for i in range(7):
    times.append((time.time() + i*86400) - base_time)

#create a dictionary to store future crimes
fut_week_crimes = dict()

fut_week_crimes_list = []
for time_i in times:
    for beat_id in beat_mapper.hash_to_key:
        temp_dict = {
            'date' : time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_i+base_time)),
            'beat': beat_mapper.get_key(beat_id),
            'type': type_mapper.get_key(clf.predict([time_i, beat_id])[0])
            }
        fut_week_crimes_list.append(temp_dict)
fut_week_crimes["crimes"] = fut_week_crimes_list
with open('future.json', 'w') as fl:
    json.dump(fut_week_crimes, fl)

print 'time to complete: %ds' % (time.time() - start_time)





yahoo.com
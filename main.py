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
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT FROM HPDCrimes ORDER BY Year ASC')

all_data_from_sql = []

for crime in db_cur.fetchall():
    temp_list = []
    temp_list.extend([crime[0], crime[1], crime[2], crime[3]])
    #convert the OffenseTypes and Beats to hashes, use a scaling coefficient
    temp_list.append(beat_mapper.get_hash(crime[4]))
    temp_list.append(type_mapper.get_hash(crime[5]))
    #append this temporary list to the main list which stores all the data
    all_data_from_sql.append(temp_list)

major_array = np.vstack(all_data_from_sql)
split_major_array = np.hsplit(major_array, 6)

X_data = np.hstack((split_major_array[0], split_major_array[1], split_major_array[2], split_major_array[3], split_major_array[4]))
y_data = np.ravel(split_major_array[5])

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

    if ctr == 0:
        plt.plot(X_test.transpose()[0], predicted, 'b')
        plt.plot(X_test.transpose()[0], y_test, 'r')
        plt.axis([0, 1.5e9, 0, 7])
        plt.show()
        ctr += 1

clf.fit(X_data[:int(len(y_data)*.7)], y_data[:int(len(y_data)*.7)])

print("Mean(accuracy_rates) = %.5f" % (np.mean(accuracy_rates)))

#reverse hash lookup

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
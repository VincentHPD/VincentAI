"""
Copyright (c) 2015 Rakshak Talwar
"""

import math, os, pdb, time, datetime, logging
from threading import Thread
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cross_validation import KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.learning_curve import learning_curve
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import vincent
start_time = time.time()

# import data from the mongo database
mongo_db_name = 'vincentdb'
crime_col_name = 'crime_instances'
mon_conn = MongoClient('localhost', 27017)  # create mongo connection
mon_db = mon_conn[mongo_db_name]  # connect to the mongo database
mon_col = mon_db[crime_col_name]  # pull up crime collection

# find the earliest and latest datetime objects
earliest_date = mon_col.find_one(
    {}, {'date': 1, '_id': 0}, sort=[("date", 1)])['date']
latest_date = mon_col.find_one(
    {}, {'date': 1, '_id': 0}, sort=[("date", -1)])['date']
# create an array to store all possible incidences of crime whether they
# occurred or not
range_dates = pd.date_range(earliest_date, latest_date, freq='D')

# create mapping objects and fill them up
type_mapper = vincent.Mapper()
beat_mapper = vincent.Mapper()
[type_mapper.get_hash(val) for val in mon_col.distinct("type_crime")]
[beat_mapper.get_hash(val) for val in mon_col.distinct("beat")]
#beat_mapper_hashes = beat_mapper.hash_to_key.keys()
#type_mapper_hashes = type_mapper.hash_to_key.keys()

# find all crime instances from mongo collection
crime_dicts = {dic["_id"]: dic for dic in mon_col.find({})}

no_crime_dicts = {}  # stores entries for combinations where no crimes occured
# find all combinations where a crime didn't occur
for date_ in range_dates:
    for beat in beat_mapper.key_to_hash.keys():
        for crime_type in type_mapper.key_to_hash.keys():
            _id = '{}-{}-{}-{}-{}'.format(date_.year,
                                          date_.month, date_.day, beat, crime_type)
            temp_dict = {
                _id:
                {'_id': _id,
                 'date': date_,
                 'beat': beat,
                 'type_crime': crime_type,
                 'n_offenses': 0}}
            # check to see if this combination already has a crime associated
            if not crime_dicts.has_key(_id):
                # if not, add it as a no crime occurrence (n_offenses = 0)
                no_crime_dicts.update(temp_dict)
# combine the data of all crime and no-crime occurences
major_data_dict = crime_dicts.copy()
major_data_dict.update(no_crime_dicts)

# make a nested list containing all of the data as dicts
xy_list = []
for key in major_data_dict:
    # extract day of week along with other date related features
    year, month, m_day, w_day = major_data_dict[key]['date'].year, major_data_dict[key][
        'date'].month, major_data_dict[key]['date'].day, major_data_dict[key]['date'].weekday()
    beat_hash = beat_mapper.get_hash(major_data_dict[key]['beat'])
    type_hash = type_mapper.get_hash(major_data_dict[key]['type_crime'])
    n_offenses = major_data_dict[key]['n_offenses']

    #DEBUG checking to see if limiting to only 3 classes will improve f1_score
    if n_offenses > 2:
        n_offenses = 2

    xy_list.append(
        {'year': year,
         'month': month,
         'm_day': m_day,
         'w_day': w_day,
         'beat_hash': beat_hash,
         'type_hash': type_hash,
         'n_offenses': n_offenses})

# make a pandas DataFrame of the data
df = pd.DataFrame(xy_list, columns=[
                  'year', 'month', 'm_day', 'w_day', 'beat_hash', 'type_hash', 'n_offenses'], dtype=int)

# we need to vectorize the beats, remove the current beat data column, and
# then concatenate the vectorized data
beats_dummies = pd.get_dummies(df['beat_hash'])
df = df.drop('beat_hash', axis=1)
df = pd.concat([df, beats_dummies], axis=1)

# now we need to vectorize the types of crime, remove the current type data column, and
# then concatenate the vectorized data
types_dummies = pd.get_dummies(df['type_hash'])
df = df.drop('type_hash', axis=1)
df = pd.concat([df, types_dummies], axis=1)

# seperate the features from the target to make X and y
n_off = df['n_offenses']  # column from the matrix which will be target values
df = df.drop('n_offenses', axis=1)  # drop the target value
X_data = df.values  # feature values
y_data = n_off.values  # target values


# create feature scaler
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#X_data_scaled = scaler.fit_transform(X_data)
X_data_scaled = X_data


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_data_scaled, y_data, test_size=0.4, random_state=42)

for pl, nn in enumerate(range(5, 10)):
    # create the classifier
    clf = KNeighborsClassifier(nn)

    # fit the algorithm
    clf.fit(X_train, y_train)

    # plotting learning curves
    plt.subplot(3, 3, pl)
    plt.title('Learning Curves for KNN k={} with Vect. Type and Beat'.format(nn))
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_data_scaled, y_data, train_sizes=np.array([.1, .2, .5, .8, .99]))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-',
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-',
             color="g", label="Cross-validation score")
    plt.legend(loc="best")

    # DEBUG
    # print f1_scores
    y_true = y_test
    y_pred = clf.predict(X_test)
    f_sco = f1_score(y_true, y_pred, average=None)

    print('for k = {0} f_scores: {1}'.format(nn, f_sco))

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
plt.xlim([-10, 10])
plt.title('Vincent')
plt.xlabel('Reduced Feature Space')
plt.ylabel('Number of Offenses')
plt.show()
"""

"""
#cross validation
accuracy_rates = []
kf = KFold(len(y_data), n_folds = 2, shuffle=True) #create cross validation model

for train_index, test_index in kf:
	X_train, X_test = X_data_scaled[train_index], X_data_scaled[test_index]
	y_train, y_test = y_data[train_index], y_data[test_index]
	regr.fit(X_train, y_train)
	predicted = regr.predict(X_test)
	accuracy = regr.score(X_test, y_test)
	accuracy_rates.append(accuracy)
"""

print 'time to complete: %ds' % (time.time() - start_time)

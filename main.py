"""
Copyright (c) 2015 Rakshak Talwar
"""

import datetime, math, os, time
import numpy as np
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT FROM HPDCrimes ORDER BY Year ASC')

all_data_from_sql = []

for crime in db_cur.fetchall():
    temp_list = []
    temp_list.extend([crime[2], crime[3]])
    #DEBUG print '{} {} {} {} {} {}'.format(crime[0], crime[1], crime[2], crime[3], crime[4], crime[5])
    #convert the OffenseTypes and Beats to hashes, use a scaling coefficient
    #temp_list.append(beat_mapper.get_hash(crime[4]))
    temp_list.append(type_mapper.get_hash(crime[5]))
    #append this temporary list to the main list which stores all the data
    all_data_from_sql.append(temp_list)

major_array = np.asarray(np.vstack(all_data_from_sql), dtype=float)

#create feature scaler and scale the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_data_scaled = scaler.fit_transform(major_array)

#use PCA to reduce the data down to 2 dimensions
reduced_data = PCA(n_components = 2).fit_transform(X_data_scaled)

#apply clustering algorithm
kmeans = KMeans(init='k-means++', n_clusters = 3, n_init = 10).fit(reduced_data)

####Visualize results####
h = 0.2 #the step size

#plot the decision boundary
#first acquirize information required to build the graph
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#obtain labels for each point
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

#Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf() #clears the plot
#show the predictions (these show up as the colored backgrounds)
plt.imshow(Z, interpolation='nearest',\
extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

#plot the points within the dataset (actual events)
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k+', markersize=1)

#plot the centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidth=3, color='w', zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Comparing Month, month day, and type of crime')

print 'time to complete: %ds' % (time.time() - start_time)

plt.show()

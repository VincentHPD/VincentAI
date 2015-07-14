import sqlite3
import pdb
import numpy as np
import pandas as pd
import pymongo

crime_database_file = 'crime_records.db'
mongo_db_name = 'vincentdb'
mongo_collection_name = 'crime_instances'
mongo_no_crime_collection_name = 'no_crime_instances'

#create and add crime instances from the database to pandas DataFrame
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT, NOffenses INTEGER FROM HPDCrimes')

mon_conn = pymongo.MongoClient('localhost', 27017) 
mon_db = mon_conn[mongo_db_name]
mon_col = mon_db[mongo_collection_name]

"""
for crime in db_cur.fetchall():
	(year, month, m_day, w_day) = (crime[0], crime[1], crime[2], crime[3])

	beat = str(crime[4])
	type_crime = str(crime[5])

	#the number of offenses which occured (target value)
	n_offenses = int(crime[6]) 

	temp_dict = {
	    'uniq' : '{}-{}-{}-{}-{}'.format(year, month, m_day, beat, type_crime),
	    'date' : pd.datetime(year, month, m_day),
	    'beat' : beat,
	    'type_crime' : type_crime,
	    'n_offenses' : n_offenses}

	mon_col.insert_one(temp_dict)
#close the sqlite connection
db_con.close()
"""

#we need to create a collection of all possible instances where no crime was commited
mon_col_2 = mon_db[mongo_no_crime_collection_name] 

###gather some preliminary information such as earliest and latest date

#create a list of all of the dates present in the crime_instances collection
all_dates = mon_col.find({}, {'date' : 1, '_id' : 0})
sorted_dates = sorted(all_dates)
earliest_date, latest_date = sorted_dates[0], sorted_dates[-1]

pdb.set_trace()

#mon_conn.close()





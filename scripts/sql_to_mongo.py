import pdb, sqlite3, time
from threading import Thread
import numpy as np
import pandas as pd
import pymongo

start_time = time.time()

crime_database_file = 'crime_records.db'
mongo_db_name = 'vincentdb'
mongo_collection_name = 'crime_instances'

#create and add crime instances from the database to pandas DataFrame
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT, NOffenses INTEGER FROM HPDCrimes')

mon_conn = pymongo.MongoClient('localhost', 27017) 
mon_db = mon_conn[mongo_db_name]

### DELETE the collection in mongo
mon_db.drop_collection(mon_db[mongo_collection_name])

#create new collection
mon_col = mon_db[mongo_collection_name]

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

#close the connections
db_con.close()
mon_conn.close()

print '%.2f' % (time.time() - start_time)

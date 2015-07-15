import pdb, sqlite3, time
from threading import Thread
import numpy as np
import pandas as pd
import pymongo

start_time = time.time()

num_threads = 10
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

### DELETE the collections in mongo
mon_db.drop_collection(mon_db[mongo_collection_name])
mon_db.drop_collection(mon_db[mongo_no_crime_collection_name])

#create new collections
mon_col = mon_db[mongo_collection_name]
mon_col_2s = [mon_db[mongo_no_crime_collection_name] for _ in range(num_threads)] #collection of all possible instances where no crime was commited

possible_crimes = set() #all different types of crime
possible_beats = set() #all different types of beats

for crime in db_cur.fetchall():
	(year, month, m_day, w_day) = (crime[0], crime[1], crime[2], crime[3])

	beat = str(crime[4])
	type_crime = str(crime[5])

	possible_crimes.add(type_crime)
	possible_beats.add(beat)

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


###gather some preliminary information such as earliest and latest date

#create a list of all of the dates present in the crime_instances collection
all_dates_docs = mon_col.find({}, {'date' : 1, '_id' : 0})
all_dates = []
for d in all_dates_docs:
	all_dates.append(d['date'])
sorted_dates = sorted(all_dates)
earliest_date, latest_date = sorted_dates[0], sorted_dates[-1]
range_dates = pd.date_range(earliest_date, latest_date, freq='D')

#find all the unique identifier in each crime_instance
uniqs_docs = mon_col.find({}, {'uniq' : 1, '_id' : 0})
uniqs = []
for u in uniqs_docs:
	uniqs.append(u['uniq'])

#find all possible combinations of dates, beats, and crimes
#if the combination doesn't exist as a crime_instance we have to add it to the no_crime_instances collection

pdb.set_trace()

def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def create_writing_thread(date_range, mon_db_col):
	"""Finds a no_crime instance and writes it to the collection"""	
	no_crime_instances = []
	for date_ in date_range:
		for beat in possible_beats:
			for crime_type in possible_crimes:
				temp_uniq = '{}-{}-{}-{}-{}'.format(date_.year, date_.month, date_.day, beat, crime_type)
				if temp_uniq not in uniqs:
					no_crime_instances.append({
						'uniq' : temp_uniq,
						'date' : date_,
						'beat' : beat,
						'type_crime' : crime_type,
						'n_offenses' : 0})
	mon_db_col.insert(no_crime_instances)

threads = []
for ctr, date_range in enumerate((chunks(range_dates, num_threads))):
	t = Thread(target=create_writing_thread, args=(date_range, mon_col_2s[ctr],))
	threads.append(t)
	t.start()
[th.join() for th in threads]

mon_conn.close()

print '%.2f' % (time.time() - start_time)


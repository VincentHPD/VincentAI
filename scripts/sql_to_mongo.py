import sqlite3
import pdb
import pandas as pd
import numpy as np

crime_database_file = 'crime_records.db'

#create and add crime instances from the database to pandas DataFrame
db_con = sqlite3.connect(crime_database_file)
db_cur = db_con.cursor()
db_cur.execute('SELECT Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, Beat TEXT, OffenseType TEXT, NOffenses INTEGER FROM HPDCrimes')

crime_dicts = {}
for crime in db_cur.fetchall():
	(year, month, m_day, w_day) = (crime[0], crime[1], crime[2], crime[3])

	beat = crime[4]
	type_crime = crime[5]

	#the number of offenses which occured (target value)
	n_offenses = crime[6] 

	temp_dict = {
	    'date' : pd.datetime(year, month, m_day),
	    'beat' : beat,
	    'type_crime' : type_crime,
	    'n_offenses' : n_offenses}

	crime_key = '{}-{}-{}-{}-{}'.format(year, month, m_day, beat, type_crime)
	crime_dicts.update({crime_key : temp_dict})
db_con.close

df = pd.DataFrame.from_dict(crime_dicts)
df = df.T

df.to_csv('crime_records.csv')

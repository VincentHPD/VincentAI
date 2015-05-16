"""
Copyright (c) 2015 Rakshak Talwar

Takes in all the excel files and converts into a sqlite3 database
"""

import datetime, math, os, time
import multiprocessing, re, sqlite3
import xlrd
from vincent import ValidData

start_time = time.time()

data_dir = 'data'

#set timezone to Houston's timezone
os.environ['TZ'] = 'America/Chicago'
time.tzset()

def rows_in_xls(xls_file_path):
    """ Pass it the path for an excel file, returns a list of lists. With each
    inner list containing the values for that row. """
    all_rows = [] #stores all rows within one excel file of data
    try:
        #open the workbook, debug logs are ignored
        wkbk = xlrd.open_workbook(xls_file_path, logfile=open(os.devnull, 'w'))
        #open the first sheet within the workbook
        sheet = wkbk.sheet_by_index(0)
        #fill up data list
        for row_ctr in range(1, sheet.nrows):
            #capture a row
            dirty_row = sheet.row_slice(row_ctr)

            #grab date value, clean it up, and convert to eTime
            date_val = xlrd.xldate_as_tuple(dirty_row[0].value, wkbk.datemode)
            date_in_proper_tuple = time.strptime(str(date_val), "(%Y, %m, %d, 0, 0, 0)")
            date_in_sec = time.mktime(date_in_proper_tuple)

            if date_in_sec < 0.0: #toss out any bad data, we can't have a date whose value is less than 0
                continue

            #create and fill up a new inner list, this list will be apended to the larger list, all_rows.
            clean_row = [item.value for item in dirty_row]
            clean_row.pop(0) #remove the old value for the date
            clean_row.insert(0, date_in_sec) #add in the clean value of date in eTime

            #if beat and offense type are valid return the crime instance
            if checker.valid_type(clean_row[2]) and checker.valid_beat(clean_row[3]):
                all_rows.append(clean_row) #add this list to the larger list

        return all_rows
    except:
        #if the excel file can't be opened return a blank list
        return []

def multiprocessing_file_reader(file_names, n_cores):
    """ """
    def worker(file_names, out_q):
        """ """
        rows_data = []

        for file_name in file_names:
            rows_data.append(rows_in_xls(os.path.join(root, file_name)))

        out_q.put(rows_data)

    out_q = multiprocessing.Queue()
    chunksize = int(math.ceil(len(file_names) / float(n_cores)))
    procs = []
    for i in range(n_cores):
        p = multiprocessing.Process(target=worker, args=(file_names[chunksize * i:chunksize * (i+1)], out_q))
        procs.append(p)
        p.start()

    results_list = []
    for i in range(n_cores):
        results_list.extend(out_q.get())

    for p in procs: #wait until all processes finish
        p.join()

    return results_list

#create a valid data checker object
checker = ValidData()

data = [] #stores all crime data
file_names = [] #stores names of xls filenames within the data/ directory
for root, dirs, filenames in os.walk(data_dir): #iterate over files
    for f in filenames:
        file_names.append(f)

#add all the lists to one main list
[data.extend(row) for row in multiprocessing_file_reader(file_names, 4)]

db_con = sqlite3.connect('crime_records.db')

cur = db_con.cursor()

cur.execute('CREATE TABLE HPDCrimes(eTime REAL, OffenseType TEXT, Beat TEXT, Premise TEXT, BlockRange TEXT, StreetName TEXT, StreetType TEXT, NumOffenses INTEGER)')

for crime in data:
    if len(crime) < 10:
        for i in range(len(crime), 9):
            crime.append('')
        crime.insert(9, 1)
    elif type(crime[9]) != float and type(crime[9]) != int:
        crime.pop(9)
        crime.insert(9, 1)
    try:
        cur.execute('INSERT INTO HPDCrimes VALUES(?, ?, ?, ?, ?, ?, ?, ?)', (crime[0], crime[2], crime[3], crime[4], crime[5], crime[6], crime[7], crime[9]))
    except:
        pass

db_con.commit() #save to database
db_con.close() #close connection

print 'time to complete: %ds' % (time.time() - start_time)






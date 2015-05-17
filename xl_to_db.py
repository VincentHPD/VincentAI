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
            print row_ctr
            #capture a row
            dirty_row = sheet.row_slice(row_ctr)

            #grab date value, clean it up, and convert to eTime
            date_val = xlrd.xldate_as_tuple(dirty_row[0].value, wkbk.datemode)
            date_in_proper_tuple = time.strptime(str(date_val), "(%Y, %m, %d, 0, 0, 0)")

            year, month, mday, wday = date_in_proper_tuple.tm_year, date_in_proper_tuple.tm_mon, date_in_proper_tuple.tm_mday, date_in_proper_tuple.tm_wday
            #create and fill up a new inner list, this list will be apended to the larger list, all_rows.
            clean_row = [item.value for item in dirty_row]

            #we need some of the data from the row before we overwrite any data
            offense_type, beat = clean_row[2], clean_row[3]

            clean_row.pop(0) #remove the old value for the date
            clean_row.insert(0, year) #add in the year as the first value in the list
            clean_row.pop(1)
            clean_row.insert(1, month)
            clean_row.pop(2)
            clean_row.insert(2, mday)
            clean_row.pop(3)
            clean_row.insert(3, wday)
            clean_row.pop(4)
            clean_row.insert(4, offense_type)
            clean_row.pop(5)
            clean_row.insert(5, beat)

            try:
                [clean_row.pop(i) for i in range(6, len(clean_row))]
            except:
                pass

            #if beat and offense type are valid return the crime instance
            if checker.valid_type(offense_type) and checker.valid_beat(beat):
                all_rows.append(clean_row) #add this list to the larger list

        return all_rows
    except Exception as e:
        #if the excel file can't be opened return a blank list
        print str(e)
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

cur.execute('CREATE TABLE HPDCrimes(Year INTEGER, Month INTEGER, MDay INTEGER, WDay INTEGER, OffenseType TEXT, Beat TEXT)')

for crime in data:
    try:
        cur.execute('INSERT INTO HPDCrimes VALUES(?, ?, ?, ?, ?, ?)', (crime[0], crime[1], crime[2], crime[3], crime[4], crime[5]))
        print ' executing sql writing {} {} {} {} {} {}'.format(crime[0], crime[1], crime[2], crime[3], crime[4], crime[5])
    except:
        pass

db_con.commit() #save to database
db_con.close() #close connection


print 'time to complete: %ds' % (time.time() - start_time)






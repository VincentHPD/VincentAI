import numpy as np

all_data_from_sql = np.array([[2010,6,2],[2011,12,31],[2012,1,5]])

earliest_year = all_data_from_sql[:, 0].min()
#find the earliest month (1-12)
earliest_month = 1000 #setting a high default value intentionally
for row in all_data_from_sql[:, :2]:
    if (row[0] == earliest_year) and (row[1] < earliest_month):
        earliest_month = row[1]
#find the earliest day (1-31) within the earliest month within the earliest year
earliest_day = 50 #setting a high default value intentionally
for row in all_data_from_sql[:, :3]:
    if (row[0] == earliest_year) and (row[1] == earliest_month) and (row[2] < earliest_day):
        earliest_day = row[2]

latest_year = all_data_from_sql[:, 0].max()
latest_month = 0 #setting a low default value intentionally
#find the latest month (1-12) within the latest year
for row in all_data_from_sql[:, :2]:
    if (row[0] == latest_year) and (row[1] > latest_month):
        latest_month = row[1]
#find the latest day (1-31) within the latest month within the latest year
latest_day = 0 #setting a low default value intentionally
for row in all_data_from_sql[:, :3]:
    if (row[0] == latest_year) and (row[1] == latest_month) and (row[2] > latest_day):
        latest_day = row[2]

print("{} {} {}".format(earliest_year, earliest_month, earliest_day))
print("{} {} {}".format(latest_year, latest_month, latest_day))

import fmongo

crime_frame = fmongo.readMongo("vincentdb", "crime_instances", query = {}, queryReturn = {})

print crime_frame

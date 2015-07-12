import fmongo

crime_frame = fmongo.readMongo("fut", "crime_instances", query = {}, queryReturn = {'type_crime' : 1})

print crime_frame

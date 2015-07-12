import pandas as pd
from pymongo import MongoClient
from bson.objectid import ObjectId

# MongoDB tutorial => http://api.mongodb.org/python/current/tutorial.html
# http://stackoverflow.com/questions/16249736/how-to-import-data-from-mongodb-to-pandas

# Connect to MongoDB
def connectMongo(host, port, username, password, db):

	""" A util for making a connection to mongo """

	if username and password:
		mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
		conn = MongoClient(mongo_uri)
	else:
		conn = MongoClient(host, port)


	return conn[db]


def readMongo(db, collection, query={}, queryReturn=None, _limit=None, host='localhost', port=27017, username=None, password=None, no_id=True):
	""" Read from Mongo and Store into DataFrame """
	"""
		nbaFrame = readMongo("nba","players",
			query= {
				"Player": "LeBron James"
			}, queryReturn={
				"Seasons.advanced": 1
			}, no_id=False)
	"""
	# Connect to MongoDB
	db = connectMongo(host=host, port=port, username=username, password=password, db=db)

	# Make a query to the specific DB and Collection
	cursor = db[collection].find(query, queryReturn)
	if _limit:
		cursor = cursor.limit(_limit)
	# Expand the cursor and construct the DataFrame
	df =  pd.DataFrame(list(cursor))

	# Delete the _id
	if no_id:
	    del df['_id']

	return df
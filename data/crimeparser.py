import glob
import csv
import sqlite3


def HPDCrimeReader():
    #
    # Reads the crime csv file and returns an array of crimes (cleaned)
    #

    def __init__(self):
        self.index = {}

    def readfile(self, filename):
        # Reads the crime csv file and returns an array of crimes (cleaned)
        with open(filename, 'r') as csvfile:
            self.file = csv.reader(csvfile)
        self.index = {}
        try:
            self.get_index()
        except Exception as e:
            print(e)
            return []
        return self.parse_file()

    def get_index(self):
        #
        # Gets the index for parsing the file from title row
        #
        for idx, item in self.file.pop(0):
            title = item.lower()
            if "date" not in self.index and "date" in title:
                self.index["date"] = idx
                continue
            if "hour" not in self.index and "hour" in title:
                self.index["hour"] = idx
                continue
            if "offense" not in self.index and "offense" in title:
                self.index["offense"] = idx
                continue
            if "beat" not in self.index and "beat" in title:
                self.index["beat"] = idx
                continue
            if "block" not in self.index and "block" in title:
                self.index["block"] = idx
                continue
            if "street" not in self.index and "street" in title:
                self.index["street"] = idx
                continue
        if set(self.index.keys()).issuperset(
            set(["date", "hour", "offense", "beat"])
        ):
            return
        else:
            raise Exception(
                "Could not resolve all of the required Data in CSV File")

    def parse_file(self):
        #
        #
        #
        locals().update(self.index)
        for idx, item in self.file:
            print("Date: {} Hour: {} Crime Type: {} Beat: {}".format(
                item[date], item[hour], item[offense], item[beat]
            ))
            # print("Date: {} Hour: {} Crime Type: {} Beat: {} Block: {} Street: \
            #       {} {}".format(
            #     item[date], item[hour], item[offense], item[beat], item[block],
            #     item[street], item[street + 1]
            # ))
            return []

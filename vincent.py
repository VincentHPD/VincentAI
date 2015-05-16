"""
Copyright (c) 2015 Rakshak Talwar
"""
import re

class Mapper():
    def __init__(self):
        self.key_to_hash = {};
        self.hash_to_key = {};

    def get_hash(self, key):
        if key not in self.key_to_hash:
            hash_val = len(self.key_to_hash)
            self.key_to_hash[key] = hash_val
            self.hash_to_key[hash_val] = key
        return self.key_to_hash[key]

    def get_key(self, hash_val):
        if hash_val in self.hash_to_key:
            return self.hash_to_key[hash_val]

class ValidData():
    """Uses regex or a list to check if data is valid"""
    def __init__(self):
        self.types = ["Aggravated Assault", "Auto Theft", "Burglary", "Murder", "Rape", "Robbery", "Theft"]
        self.beat_pattern = '\d?\d[A-Z]\d{2}' #checks for a valid beat pattern. Which is 1-2 digits followed by an uppercase letter and finally followed by 2 digits
        self.beat_matcher = re.compile(self.beat_pattern)

    def valid_type(self, p_type):
        """ If p_type is a valid type; return True. Return False otherwise. """
        if p_type in self.types:
            return True
        else:
            return False

    def valid_beat(self, p_beat):
        """ If p_beat is in the valid beat format; return True. False otherwise. """
        if self.beat_matcher.match(p_beat):
            return True
        else:
            return False

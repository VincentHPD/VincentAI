"""
Copyright (c) 2015 Rakshak Talwar
"""

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


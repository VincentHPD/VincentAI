import pandas as pd

many_d = {'_i': {'val2': 2, 'val1': 1}, '_i2': {'val2': 4, 'val1': 3}}

many_d_ls = [many_d[key] for key in many_d]

print many_d_ls

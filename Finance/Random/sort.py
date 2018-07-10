import pprint
import csv
import operator
import numpy as np
import matplotlib.pyplot as plt

#############################

with open('AAPL.csv', 'rb') as f:
	reader = csv.reader(f)
	data = list(reader)

data.sort(key=lambda tup: tup[1])
pprint.pprint(data)



1 -4 2 -5 2.25 -

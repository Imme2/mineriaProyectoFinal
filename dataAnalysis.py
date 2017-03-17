import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import cPickle
import numpy as np

# (faltan imports)

messages = pandas.read_csv('DATA_TAB/train_3000.label', sep="\t",quoting=csv.QUOTE_NONE,
							names=["label","sublabel","question"])

messages['length'] = messages['question'].map(lambda text: len(text))

messages.hist(column='length',by='label',bins = 40)
plt.show()

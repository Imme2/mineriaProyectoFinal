import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import numpy as np

# (faltan imports)

messages = pandas.read_csv('DATA_TAB/train_2000.label', sep="\t",quoting=csv.QUOTE_NONE, encoding = "ISO-8859-1",
							names=["label","sublabel","question"])

messages['length'] = messages['question'].map(lambda text: len(text))

messages.hist(column='length',by='label',bins = 40)
plt.show()

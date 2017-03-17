import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import numpy as np
from collections import Counter

def split_into_tokens(message):
    return TextBlob(message).words


def return_tags(message):
    return TextBlob(message).tags

print("Reading File")
messages = pandas.read_csv('DATA_TAB/train_5500.tab', sep="\t",quoting=csv.QUOTE_NONE,encoding = "ISO-8859-1",
							names=["label","sublabel","question"])

print("Done.")

r = messages.question.apply(return_tags)
g = []
rows = []
for i,message in enumerate(r):
    labs = [ty[1] for ty in message]
    g+= labs
    new_dic = dict(Counter(labs))
    new_dic["LABEL"] = messages.ix[i].label
    rows += [new_dic]
g = list(set(g))
g.append("LABEL")

print("Generating")
h = pandas.DataFrame(columns = g)
for i,x in enumerate(rows):
    h = h.append(pandas.DataFrame(x,index=[i,],columns = g))
    pass
print("Saving")
h = h.fillna(0)
h.to_csv("tagger.csv")

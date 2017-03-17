import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def split_into_tokens(message):
    return TextBlob(message).words

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

if __name__ == '__main__':
	# (faltan imports)
	messages = pandas.read_csv('DATA_TAB/train_3000.tab', sep="\t",quoting=csv.QUOTE_NONE,
								encoding = "ISO-8859-1",
								names=["label","sublabel","question"])

	messages['length'] = messages['question'].map(lambda text: len(text))

	messages.hist(column='length',by='label',bins = 40)
	plt.savefig("Histogramas/train_3000.png")

	bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['question'])

	messages_bow = bow_transformer.transform(messages['question'])

	tfidf_transformer = TfidfTransformer().fit(messages_bow)
	messages_tfidf = tfidf_transformer.transform(messages_bow)
	# Aqui en messages_tfidf y con el transformer ya tenemos todas las palabras

	with open("TFIDF/train_3000.tfidf","w") as f:
		for word in bow_transformer.vocabulary_:
			f.write(word.encode("utf-8") + " ")
			f.write(str(tfidf_transformer.idf_[bow_transformer.vocabulary_[word]]))
			f.write("\n")
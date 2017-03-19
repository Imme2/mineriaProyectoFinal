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

	# Una version resumida del archivo
	with open("TFIDF/train_3000.tfidf2","w") as f:
		palabras = messages_tfidf.shape[1]
		preguntas = messages_tfidf.shape[0]

		for i in range(preguntas):
			for j in range(palabras):
				if (messages_tfidf[i,j] != 0):
					f.write(bow_transformer.get_feature_names()[j].encode('utf-8'))
					f.write("\t")
					f.write("("+ str(i) + ", " + str(j) + ")")
					f.write("\t")
					f.write(str(messages_tfidf[i,j]))
					f.write('\n')

	# Esto crea un archivo de 66 mb, no es recomendable correrlo.
	# with open("TFIDF/train_3000.tfidf","w") as f:
	# 	palabras = messages_tfidf.shape[1]
	# 	preguntas = messages_tfidf.shape[0]

	# 	f.write(bow_transformer.get_feature_names()[0].encode('utf-8'))
	# 	for i in range(1,messages_tfidf.shape[1]):
	# 		f.write('\t' + bow_transformer.get_feature_names()[i].encode('utf-8'))

	# 	f.write('\n')
	# 	for i in range(preguntas):
	# 		f.write(str(messages_tfidf[i,0]))
	# 		for j in range(1,palabras):
	# 			f.write('\t' + str(messages_tfidf[i,j]))
	# 		f.write('\n')
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


	messages = pandas.read_csv('DATA_TAB/train_1000.tab', sep="\t",quoting=csv.QUOTE_NONE,
								encoding = "ISO-8859-1",
								names=["label","sublabel","question"])


	bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['question'])

	grupos = messages.groupby('label').apply(lambda x: x.sum())
	messages_bow_grupos = bow_transformer.transform(grupos['question'])
	tfidf_transformer_grupos = TfidfTransformer().fit(messages_bow_grupos)
	messages_tfidf_grupos = tfidf_transformer_grupos.transform(messages_bow_grupos)


	clases = messages_tfidf_grupos.shape[0]
	palabras = messages_tfidf_grupos.shape[1]

	for i in range(clases):
		plt.title(messages['label'].unique()[i])
		plt.xlabel('palabras')
		plt.ylabel('tf-idf')
		auxHist = []
		for j in range(palabras):
			auxHist += [messages_tfidf_grupos[i,j]]
		enum = [k for k in range(len(auxHist))]
		plt.hist(enum,bins=palabras/4,weights=auxHist)
		plt.savefig("Histogramas/histogramaTFIDF_" + messages['label'].unique()[i])

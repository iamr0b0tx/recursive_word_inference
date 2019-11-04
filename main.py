# from the lib code
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordnetwork import WordNetwork
from functions import *

import os
import numpy as np

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.datasets import fetch_20newsgroups

def runLDA(train_docs):
	# Initialise the count vectorizer with the English stop words
	count_vectorizer = CountVectorizer(stop_words='english')

	# Fit and transform the processed titles
	lda_vector = count_vectorizer.fit_transform(train_docs)

	# Helper function
	def print_topics(model, count_vectorizer, n_top_words):
		words = count_vectorizer.get_feature_names()
		for topic_idx, topic in enumerate(model.components_):
			print("Topic #{}: {}".format(topic_idx, " ".join([words[i]for i in topic.argsort()[:-n_top_words - 1:-1]])))
		print()

	# Tweak the two parameters below
	number_topics = 3
	number_words = 10

	# Create and fit the LDA model
	lda = LDA(n_components=number_topics)
	lda.fit(lda_vector)

	# Print the topics found by the LDA model
	print("Topics found via LDA:\n===========================")
	print_topics(lda, count_vectorizer, number_words)

	return lda, lda_vector

def main():
	# load wordnet model
	WN = WordNetwork()
	
	# the training folder
	training_folder = '../min_train'

	# get the docs/corpus
	# train_docs = getDFData('../data/full-corpus.csv', 'TweetText')
	# train_docs = getDFData('../data/papers.csv', 'abstract')

	# train_docs = getData(training_folder)['text']
	# test_docs = [read_txt('{}/{}'.format(training_folder, filename)) for filename in os.listdir(training_folder)]

	ntrain = 10
	ntest = 1#int(ntrain * 0.1)

	train_docs = fetch_20newsgroups(subset='train', shuffle=True)
	train_docs, train_docs_target, classes = train_docs.data[:ntrain], train_docs.target[:ntrain], train_docs.target_names

	# test_docs = fetch_20newsgroups(subset='test', shuffle=True)
	# test_docs, test_docs_target, classes = test_docs.data[:ntest], test_docs.target[:ntest], test_docs.target_names
	test_docs, test_docs_target, classes = train_docs, train_docs_target, classes

	# the word net model training
	WN.train(train_docs)

	# the lda model
	lda, lda_vector = runLDA(train_docs)
	
	print('Classification\n=====================================')
	ntd, docs = [], []
	for doc_i, doc_text in enumerate(test_docs):
		# the result
		topics, words_in_doc = WN.classifyDoc(doc_text)
		ntd.append(list(topics))
		docs.append(words_in_doc)

		top_topics = topics.sort_values(ascending=False)[:3]
		top_topics = (top_topics.index, top_topics.values)
		
		# show
		print('doc {}: topic = {}, class = {}'.format(doc_i, top_topics, classes[test_docs_target[doc_i]]))
	print()

	# nummber of topic to docs
	ntd = np.array(ntd).T

	# lda perplexity
	lda_perplexity = lda.perplexity(lda_vector)

	# WordNetwork perplexity
	wn_perplexity = perplexity(docs, WN.topic_word_distr.T, ntd)

	# display the perplexity
	print(f'Perplexity\n==============\nlda = {lda_perplexity:.4f}, word_network = {wn_perplexity:.4f}')

if __name__ == '__main__':
	main()

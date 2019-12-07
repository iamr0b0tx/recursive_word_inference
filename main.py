# from the lib code
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordnetwork import WordNetwork
from functions import *

import os
import numpy as np

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.simplefilter("ignore", DeprecationWarning)


def runLDA(train_docs, test_docs, number_topics):
	# Initialise the count vectorizer with the English stop words
	count_vectorizer = CountVectorizer(stop_words='english')

	# Fit and transform the processed titles
	lda_vector = count_vectorizer.fit_transform(train_docs)

	# Helper function
	def print_topics(model, count_vectorizer, n_top_words):
		words = count_vectorizer.get_feature_names()
		for topic_idx, topic in enumerate(model.components_):
			print("Topic #{}: {}".format(topic_idx, " ".join(
				[words[i]for i in topic.argsort()[:-n_top_words - 1:-1]])))
		print()

	# Tweak the two parameters below
	number_words = 10

	# Create and fit the LDA model
	lda = LDA(n_components=number_topics)
	lda.fit(lda_vector)

	# Print the topics found by the LDA model
	print("Topics found via LDA:\n===========================")
	print_topics(lda, count_vectorizer, number_words)

	return lda, lda_vector, CountVectorizer(stop_words='english').fit_transform(test_docs)


def main():
	# load wordnet model
	WN = WordNetwork()

	ntrain = 15
	ntest = 1 # int(ntrain * 0.1)
	shuffle_state = False

	train_docs = fetch_20newsgroups(
		subset='train', shuffle=shuffle_state, remove=('headers', 'footers', 'quotes')
	)
	train_docs, train_docs_target, classes = train_docs.data[:ntrain], train_docs.target[:ntrain], train_docs.target_names

	# test_docs = fetch_20newsgroups(subset='test', shuffle=shuffle_state, remove=('headers', 'footers', 'quotes'))
	# test_docs, test_docs_target, classes = test_docs.data[:ntest], test_docs.target[:ntest], test_docs.target_names
	test_docs, test_docs_target, classes = train_docs, train_docs_target, classes

	# redifine classes
	all_classes = classes.copy()
	def first(x): return x.split('.')[0]
	# first = lambda x: x
	classes = list(set([first(x) for x in classes]))
	class_indices = {
		i: classes.index(first(x)) for i, x in enumerate(all_classes)
	}

	# the word net model training
	WN.train(train_docs)

	# the num of topics discovered
	number_of_topics = len(WN.topic_word_distr.columns)
	
	# the shape of the confusionmatrix
	shape = (number_of_topics, len(classes))

	# the lda model
	lda, lda_vector, test_vector = runLDA(train_docs, test_docs, number_of_topics)

	print('Classification\n=====================================')
	ntd, docs = [], []
	confusion_matrix = np.zeros(shape)

	# lda confusion matrix
	lda_topic_word_distr = lda.transform(test_vector)
	lda_confusion_matrix = np.zeros(shape)

	for doc_i, doc_text in enumerate(test_docs):
		# the result
		topics, words_in_doc = WN.classifyDoc(doc_text)
		ntd.append(list(topics.values))
		docs.append(words_in_doc)

		# classification
		actual_class = all_classes[test_docs_target[doc_i]]
		class_index = class_indices[test_docs_target[doc_i]]
		class_ = classes[class_index]

		top_topics = topics.sort_values(ascending=False)[:3]
		top_topics = (top_topics.index, top_topics.values)
		confusion_matrix[top_topics[0][0]][class_index] += 1
		lda_confusion_matrix[lda_topic_word_distr[doc_i].argsort()[-1]][class_index] += 1

		# show
		print('doc {}: topic = {}, class = {}: {}'.format(
			 doc_i, top_topics, class_, actual_class)
		)
	print()

	print('network_clusters = {}, lda_clusters = {}, topics = {}\n'.format(
		len(WN.topic_word_distr.columns), lda_topic_word_distr.shape[1], classes)
	)

	#calculate the lda purity
	lda_purity = lda_confusion_matrix.max(1).sum() / len(test_docs)

	# the purity value
	purity = confusion_matrix.max(1).sum() / len(test_docs)
	print(
		f'Purity\n==============\nlda = {lda_purity:.4f}, word_network = {purity:.4f}\n')

	# initilaize entropy
	entropy = lda_entropy = 0

	for topic in WN.topic_word_distr.columns:
		topic_sum = confusion_matrix[topic].sum()
		lda_topic_sum = lda_confusion_matrix[topic].sum()
		
		lda_H_w = H_w = 0
		for class_index in range(len(classes)):
			x = confusion_matrix[topic][class_index] / topic_sum if topic_sum > 0 else 0
			lda_x = lda_confusion_matrix[topic][class_index] / lda_topic_sum

			# final entropy for cluster
			H_w += x * log(x, 2)
			lda_H_w += lda_x * log(lda_x, 2)
		
		# entropy
		entropy += -H_w * (topic_sum / confusion_matrix.size)
		lda_entropy += -lda_H_w * (lda_topic_sum / lda_confusion_matrix.size)

	#display the entropy
	print(f'Entropy\n==============\nlda = {lda_entropy:.4f}, word_network = {entropy:.4f}\n')

	# get the coherence for the topics
	coherence, all_coherences = WN.getCoherence()
	lda_coherence = 0

	# display the coherence of topics	
	print(f'Coherence\n==============')
	for topic, topic_coherence in enumerate(all_coherences):
		print(f'  topic = {topic}: lda = {lda_coherence:.4f}, word_network = {topic_coherence:.4f}')
	
	print(f'lda = {lda_coherence:.4f}, word_network = {coherence:.4f}\n')

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

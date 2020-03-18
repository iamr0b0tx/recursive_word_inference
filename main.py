# from the lib code
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordnetwork import WordNetwork
from utility import *

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

	return lda, lda_vector, count_vectorizer.transform(test_docs)


def main():
	# load wordnet model
	WN = WordNetwork()

	ntrain = 300
	ntest = int(ntrain * 0.5)
	# ntest = ntrain  # int(ntrain * 0.5)
	shuffle_state = False

	# retrieve dataset
	train_docs = fetch_20newsgroups(subset='train', shuffle=shuffle_state, remove=('headers', 'footers', 'quotes')	)
	train_docs, train_docs_target, classes = train_docs.data[:ntrain], train_docs.target[:ntrain], train_docs.target_names

	test_docs = fetch_20newsgroups(subset='test', shuffle=shuffle_state, remove=('headers', 'footers', 'quotes'))
	test_docs, test_docs_target, classes = test_docs.data[:ntest], test_docs.target[:ntest], test_docs.target_names
	# test_docs, test_docs_target, classes = train_docs, train_docs_target, classes

	# class modifier
	def modify_class(var):
		# return var
		return var.split('.')[0]

	# the output topic
	train_docs_target = [modify_class(classes[ci]) for ci in train_docs_target]
	test_docs_target = [modify_class(classes[ci]) for ci in test_docs_target]

	# the word net model training
	WN.train(train_docs, train_docs_target)

	# the topics used for modelling
	all_topics = list(set([modify_class(cl) for cl in classes]))

	# the num of topics discovered
	number_of_topics = len(all_topics)
	
	# the lda model
	lda, lda_vector, test_vector = runLDA(train_docs, test_docs, number_of_topics)

	print('Classification\n=====================================')

	# the network confusion matrix
	ntd, docs = [], []
	confusion_matrix = pd.DataFrame(
		data=0.0,
		columns=all_topics,
		index=all_topics
	)

	# lda confusion matrix
	lda_topic_word_distr = lda.transform(test_vector)
	lda_confusion_matrix = pd.DataFrame(
		data=0.0,
		columns=all_topics,
		index=all_topics
	)

	# the accuracy
	acc1 = acc2 = 0

	for doc_i, doc_text in enumerate(test_docs):
		# the result
		topics, words_in_doc = WN.classifyDoc(doc_text)
		ntd.append(list(topics.values))
		docs.append(words_in_doc)

		# classification
		class_index = test_docs_target[doc_i]

		top_topics = topics.sort_values(ascending=False)[:3]
		top_topics = list(zip(list(top_topics.index), list(top_topics.values)))
		confusion_matrix[top_topics[0][0]][class_index] += 1

		if top_topics[0][0] == class_index:
			acc1 += 1
			acc2 += 1

		else:
			if top_topics[1][0] == class_index:
				acc2 += 0.5

		lda_top_topics = lda_topic_word_distr[doc_i].argsort()[::-1]
		lda_confusion_matrix[all_topics[lda_top_topics[0]]][class_index] += 1

		topic_info = ", ".join([
			f"{topic:5s} = {topic_value:.4f}" for topic, topic_value in top_topics
		]).strip()

		# show
		print(f'doc {doc_i}: topic = [{topic_info}, lda = {lda_top_topics[0]}], class = {class_index}')
	print()

	# calculate test classification accuracy
	acc1 /= ntest
	acc2 /= ntest

	print('network_clusters = {}, lda_clusters = {}, topics = {}, acc1 = {}, acc2 = {}\n'.format(
			len(WN.topic_word_distr.columns), lda_topic_word_distr.shape[1], all_topics, acc1, acc2
		)
	)

	#calculate the lda purity
	lda_purity = lda_confusion_matrix.max(1).sum() / len(test_docs)

	# the purity value
	purity = confusion_matrix.max(1).sum() / len(test_docs)
	print(
		f'Purity\n==============\nlda = {lda_purity:.4f}, word_network = {purity:.4f}\n')

	# calculate entropy
	x = confusion_matrix.sum(0) / ntest
	lda_x = lda_confusion_matrix.sum(0) / ntest

	entropy = (-x * x.apply(lambda value: log(value, 2))).sum()
	lda_entropy = (-lda_x * lda_x.apply(lambda value: log(value, 2))).sum()

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

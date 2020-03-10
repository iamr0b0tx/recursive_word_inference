# import fron the python stdlib
import re
import os

# import from third party lib
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

from tqdm import tqdm
from nltk import word_tokenize
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

# from lib code
from utility import console_log, log

# stopwords
cachedStopWords = stopwords.words("english")


class WordNetwork:
	def __init__(self):
		pass

	def classifyDoc(self, doc_text):
		doc_word_distr = pd.DataFrame(
			data=0.0,
			columns=[0],
			index=self.topic_word_distr.index
		)[0]

		tokens = self.tokenize(doc_text)
		words_in_doc = []

		for word in tokens:
			if word not in self.topic_word_distr.index:
				continue

			# save word in doc
			words_in_doc.append(word)

			# use latest model of term ratio to develop doc word relation model
			doc_word_distr += self.term_term_ratio[word]

		# make words unique
		words_in_doc = list(set(words_in_doc))

		# create topic word distribution
		N = len(words_in_doc)
		doc_word_distr = doc_word_distr / N if N > 0 else doc_word_distr * 0
		doc_word_distr = doc_word_distr.T

		# topic word distr temp
		topic_distr = pd.DataFrame(
			data=0.0,
			columns=[0],
			index=self.topic_word_distr.columns
		)[0]

		# the relation between word and topic
		word_topic_distr = self.topic_word_distr.T

		# doc topic word distr
		for word in words_in_doc:
			topic_distr += doc_word_distr[word] * word_topic_distr[word]
			# print(topic_distr)
			# print(doc_word_distr[word])
			# print(self.topic_word_distr.T[word])
			# print()

		# the final result of relations
		N = len(words_in_doc)
		topic_distr = topic_distr / N if N > 0 else topic_distr * 0
		# topic_distr = topic_distr[0]

		# console_log('{}{} Documnet Topic (probability) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		# console_log(topic_distr, end='\n\n')

		# self.topic_word_distr.to_csv('twd.csv')
		# doc_word_distr.to_csv('dwd.csv')

		# self.getTopWords(topic_word_distr=topic_word_distr)

		return topic_distr, words_in_doc

	def get_clusters2(self, values):
		cluster_ends = np.linspace(min(values), max(values), num=7)
		values = np.array(values)
		value_indices = np.array(range(values.size))

		topics = []
		for index, end in enumerate(cluster_ends[1:]):
			topics.append(
				list(value_indices[(cluster_ends[index-1] <= values) & (values <= end)]))
		print(topics)
		return topics

	def get_clustersx(self, values):
		d = diff = np.diff(values)
		# d = diff[diff > 0]
		diff_mean = d.mean()

		last_index = 0
		topic, topics = [0], []
		diff_last_index = len(diff) - 1

		# loop through the data to accumulate the topics
		for i, x in enumerate(diff):
			index = i+1

			# if the mean difference is in range of known then add it to topic
			if x < diff_mean:
				if last_index + 1 == index:
					topic.append(index)

				else:
					if len(topic) > 0:
						topics.append(topic)
					topic = [index]

				# if this index is the last wrap up the topic
				if i == diff_last_index:
					topics.append(topic)

			# if not save as an independent topic
			else:
				if len(topic) > 0:
					topics.append(topic)

				topic = [index]
				if i == diff_last_index:
					topics.append(topic)

			# console_log(x, index, last_index, topic)
			last_index = index

		return topics

	def getClusters(self, topic_word_distr, algorithm=None):
		if algorithm is None:
			res = topic_word_distr.mean(0)
			sorted_res = sorted(res)

			indices = np.array(np.argsort(res))
			topics = self.get_clusters(sorted_res)

			# console_log(indices, topics)
			console_log('{}{} {} topic(s) found {}{}'.format(
				' '*25, '='*25, len(topics), '='*25, ' '*25))

			tp = []
			for topic in topics:
				indx = indices[topic]

				tp.append([])
				for index in indx:
					tp[-1].append(index)
					# console_log(model.columns[index], round(res[index], 4))
				# console_log()

			return tp

		elif algorithm == 'kmeans':
			X = self.doc_term_freq
			kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,
							n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
							random_state=None, tol=0.0001, verbose=0)

			kmeans.fit(X)
			print(kmeans.inertia_)
			input()

			return

	def constructDTF(self, docs, labels):
		# construct the model
		doc_term_freq = {}
		doc_topic = {}
		term_term_freq = {}
		term_term_ratio = {}

		# docs that word belongs
		word_docs = {}

		console_log('-'*50, 'Building Document Term Matrix!', '-'*50)

		# build vocabulary
		for doc_index in tqdm(range(len(docs))):
			if doc_index not in doc_term_freq:
				doc_term_freq[doc_index] = {}
				doc_topic[doc_index] = {}

			text = docs[doc_index]
			doc_topic[doc_index][labels[doc_index]] = 1

			# get word tokens
			tokens = self.tokenize(text)

			for token in tokens:
				if token not in doc_term_freq[doc_index]:
					doc_term_freq[doc_index][token] = 0

				doc_term_freq[doc_index][token] += 1

				# check if token already initialized in word_doc
				if token not in word_docs:
					word_docs[token] = []

				# add the doc that word belong
				word_docs[token].append(doc_index)
		console_log()

		console_log('-'*50, 'Building Word Occurrence and Co-occurrence!', '-'*50)
		for token1 in tqdm(word_docs):
			wd1 = word_docs[token1]
			if token1 not in term_term_freq:

				term_term_freq[token1] = {}
				term_term_ratio[token1] = {}

			for token2 in word_docs:
				wd2 = word_docs[token2]

				term_term_freq[token1][token2] = len(set(wd1).intersection(set(wd2)))
				term_term_ratio[token1][token2] = term_term_freq[token1][token2] / len(wd1) if len(wd1) > 0 else 0

		# make a dataframe
		self.doc_topic = pd.DataFrame(doc_topic)
		self.doc_term_freq = pd.DataFrame(doc_term_freq)

		# term term matrix
		term_term_freq = pd.DataFrame(term_term_freq)
		term_term_ratio = pd.DataFrame(term_term_ratio)

		# set word_docs as field
		self.word_docs = word_docs.copy()

		# replace nan as 0
		self.doc_topic.fillna(0, inplace=True)
		self.doc_term_freq.fillna(0, inplace=True)
		term_term_freq.fillna(0, inplace=True)
		term_term_ratio.fillna(0, inplace=True)

		# adjust term ratio with trust factor
		self.term_term_ratio = self.trustFactor(term_term_freq) * term_term_ratio

		console_log()
		console_log(self.doc_topic, '\n')

	def getCoherence(self, topwords=None):
		# get the topwords
		if type(topwords) == int:
			topwords = self.getTopWords(num=topwords, verbose=False)

		elif topwords is None:
			topwords = self.getTopWords(verbose=False)

		# initialized coherence
		coherence = 0
		all_coherences = []

		for topic_topwords in topwords:
			# get all npmi in topic
			topic_npmis = [self.npmi(w1, w2)
						   for w1 in topic_topwords for w2 in topic_topwords]
			topic_coherence = sum(topic_npmis) / \
				len(topic_npmis) if len(topic_npmis) > 0 else 0

			# gather all coherence
			all_coherences.append(topic_coherence)
			coherence += topic_coherence

		coherence /= len(topwords) if len(topwords) > 0 else 1
		return coherence, all_coherences

	def getTopWords(self, verbose=1, num=10, topic_word_distr=None):
		if topic_word_distr is None:
			topic_word_distr = self.topic_word_distr

		topwords = []
		# geting the top words influencing topic
		for t in topic_word_distr.columns:
			topic_topwords = topic_word_distr[t].sort_values(ascending=False)[
				:num]

			if verbose:
				console_log('Topic {}\n{}'.format(t, '='*10))
				console_log(topic_topwords, end='\n\n')

			topwords.append(list(topic_topwords.index))

		return topwords

	def npmi(self, word1, word2):
		pmi = log(self.term_term_ratio[word1][word2] / (
			self.term_term_ratio[word1].sum() * self.term_term_ratio[word2].sum()))
		denom = log(self.term_term_ratio[word1][word2])
		return pmi / -(denom) if denom > 0 else 0

	def run_iteration(self):
		term_doc_freq = self.doc_term_freq.T
		term_topic = pd.DataFrame(data=0, columns=term_doc_freq.columns, index=self.doc_topic.index)

		console_log('\n', '-'*50, 'Constructing Topic word distribution!', '-'*50)

		for word in tqdm(term_doc_freq.columns):
			tdf = term_doc_freq[word]  # word term_doc_freq

			doc_indices = tdf[tdf > 0].index
			term_topic[word] += self.doc_topic[doc_indices].sum(1)

		# leave a line
		console_log()

		# the general topic word matrix
		word_topic_matrix = term_topic.copy()

		console_log('\n', '-'*50, 'Checking for informative words!', '-'*50)

		# non informative columns
		columns_to_drop = []

		# check for informative words
		for term in tqdm(term_topic.columns):
			# normalize the term to topic
			word_topic_matrix[term] /= word_topic_matrix[term].sum()

			# select the ones with diluted topics to drop
			# if not (term_topic[term] == term_topic[term].sum()).any():
			# if not (term_topic[term] > term_topic[term].mean()).any():
			if not (term_topic[term] > term_topic[term].mean() + term_topic[term].min()).any():
				columns_to_drop.append(term)

		# flip the word topic to get topic word
		self.topic_word_distr = word_topic_matrix.T

		# display the topic-word distr
		console_log(self.topic_word_distr.T)

		# leave a line
		console_log()

		# drop non informative columns
		term_topic = term_topic.drop(columns=columns_to_drop)

		# trim down the informative words
		best_words_indices = []
		topic_term = term_topic.T

		console_log('\n', '-'*50, 'Trimming for informative words!', '-'*50)

		for topic in tqdm(term_topic.index):
			topic_terms = topic_term[topic]
			topic_terms_mean = np.unique(topic_terms.values).mean()

			best_words_indices.extend(
				list(
					topic_terms[topic_terms > topic_terms_mean].index
				)
			)

		# leave a line
		console_log()

		# the unique index of the best words
		best_words_indices = list(set(best_words_indices))

		# display topwords for topi word distr
		# self.getTopWords(topic_word_distr=self.topic_word_distr.T[best_words_indices].T)

		if 1:
			# the new term term ratio to be infered from best of best
			# temp term term ratio matrix
			ttr = self.term_term_ratio[best_words_indices] * 0

			# display the current runing process
			console_log('\n', '-'*50, 'Infering best_word-word ratio!', '-'*50)

			# infer word for word
			for w1 in tqdm(best_words_indices):
				for w2 in best_words_indices:
					factor = 1

					# inference from sharing occurence with informative word
					# co_occurence_inference_factor = self.term_term_ratio[w1][w2]
					# if co_occurence_inference_factor > 0:
					# 	factor *= co_occurence_inference_factor

					# inference from sharing topic with best word
					co_topic_inference_factor = (
						self.topic_word_distr.T[w1] * self.topic_word_distr.T[w2]).mean()
					if co_topic_inference_factor > 0:
						factor *= co_topic_inference_factor

					# infer relation of words
					ttr[w2] += factor * self.term_term_ratio[w1]

			# normalize
			self.term_term_ratio[best_words_indices] = ttr / len(best_words_indices) if len(best_words_indices) > 0 else ttr * 0
			
		if 1:
			# the new term ratio to be infered from best
			ttr = self.term_term_ratio * 0  # temp term term ratio matrix

			# display the current runing process
			console_log(
				'\n', '-'*50, 'Infering word-word ratio!', '-'*50)

			# infer word for word
			for w1 in tqdm(best_words_indices):
				for w2 in self.term_term_ratio.index:
					factor = 1

					# inference from sharing occurence with informative word
					# co_occurence_inference_factor = self.term_term_ratio[w1][w2]
					# if co_occurence_inference_factor > 0:
					# 	factor *= co_occurence_inference_factor

					# inference from sharing topic with best word
					co_topic_inference_factor = (self.topic_word_distr.T[w1] * self.topic_word_distr.T[w2]).mean()
					if co_topic_inference_factor > 0:
						factor *= co_topic_inference_factor

					# infer relation of words
					ttr[w2] += factor * self.term_term_ratio[w1]

			# normalize
			self.term_term_ratio = ttr / len(best_words_indices) if len(best_words_indices) > 0 else ttr * 0

		# console_log(term_term_ratio, '\n')
		# input('enter to continue!')

		# the most informative words
		self.best_words_indices = best_words_indices.copy()
		return

	def train(self, docs, labels, n=1.0):
		# the documents that word appears in
		docs_length = len(docs)

		# the batch to learn
		docs_length = int(n * docs_length) if type(n) == float else n

		# ---------------------------------------------preprocessing----------------------------------
		console_log('-'*30, 'Preprocessing!', '-'*30, '\n')

		# constructs a model
		self.constructDTF(docs, labels)

		# ----------------------------------------training----------------------------------------
		console_log('='*30, 'Training!', '='*30, '\n')

		# iterate and infer
		self.run_iteration()

		# set topic word_distr
		topic_word_distr = self.topic_word_distr.T

		# show the topic distribution
		console_log('{}{} Topic (word) distribution! {}{}'.format(
			' '*25, '='*25, '='*25, ' '*25))
		console_log(topic_word_distr, end='\n\n')

		# geting the percentage influence of word to topic
		# self.topic_word_distr /= topic_word_distr.sum(1)

		# console_log('{}{} Topic (word(%)) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		# console_log(self.topic_word_distr.T, end='\n\n')

		# self.topic_word_distr = topic_word_distr
		self.getTopWords()

		console_log('{} doc(s) read and {} word(s) in the vocabulary'.format(
			docs_length, len(self.word_docs)))
		return

	def trustFactor(self, x):
		if type(x) == list:
			x = len(x)
		return x / (1 + x)

	def tokenize(self, text):
		'''
		this makes use of regular expression to break the sentences into standalone words
		in this case the PortStemmer is used along side te RegExp
		'''

		# remove punctuations
		words = map(lambda word: word.lower().replace(
			"\n", '.'), word_tokenize(text))

		# remove stopwords
		words = [word for word in words if word not in cachedStopWords]

		# stemming
		# tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
		tokens = words

		# filter the lower than minimum length
		min_length = 3
		p = re.compile('[a-zA-Z]+')
		filtered_tokens = list(filter(lambda token: p.match(
			token) and len(token) >= min_length, tokens))
		# filtered_tokens = tokens

		return filtered_tokens

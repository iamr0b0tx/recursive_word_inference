# import fron the python stdlib
import re, os

# import from third party lib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

from tqdm import tqdm
from nltk import word_tokenize
# from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

# from lib code
from vocabulary import Vocabulary
from functions import console_log

# stopwords
cachedStopWords = stopwords.words("english")

class WordNetwork:
	def __init__(self):
		# processing chunks
		self.batch_size = 1000

	def classifyDoc(self, doc_text):
		doc_word_distr = pd.DataFrame(
			data=0.0, 
			columns=self.topic_word_distr.columns,
			index=self.topic_word_distr.index
		)

		tokens = self.tokenize(doc_text)
		words_in_doc = []

		for word in tokens:
			if word in self.topic_word_distr.index:
				words_in_doc.append(word)

				for topic_index in self.topic_word_distr.columns:	
					doc_word_distr[topic_index] += self.term_term_ratio[word]
				

		doc_word_distr = doc_word_distr.apply(self.trustFactor)
		topics = doc_word_distr.mean(axis=0)

		# console_log('{}{} Document (word_freq) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		# console_log(doc_word_distr, end='\n\n')

		# console_log('{}{} Documnet Topic (probability) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		# console_log(topics, end='\n\n')

		self.topic_word_distr.to_csv('twd.csv')
		doc_word_distr.to_csv('dwd.csv')

		return topics, list(set(words_in_doc))

	def get_clusters(self, values):
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
			console_log('{}{} {} topic(s) found {}{}'.format(' '*25, '='*25, len(topics), '='*25, ' '*25))

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
		
	def constructDTF(self, docs):
		# construct the model
		doc_term_freq = {}
		term_term_freq = {}
		term_term_ratio = {}

		# docs that word belongs
		word_docs = {}

		console_log('-'*50, 'Building Document Term Matrix!', '-'*50)

		# build vocabulary
		for doc_index in tqdm(range(len(docs))):
			if doc_index not in doc_term_freq:
				doc_term_freq[doc_index] = {}

			text = docs[doc_index]

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
		self.doc_term_freq = pd.DataFrame(doc_term_freq)

		# term term matrix
		self.term_term_freq = pd.DataFrame(term_term_freq)
		self.term_term_ratio = pd.DataFrame(term_term_ratio)
		
		# set word_docs as field
		self.word_docs = word_docs.copy()

		# replace nan as 0
		self.doc_term_freq.fillna(0, inplace=True)
		self.term_term_freq.fillna(0, inplace=True)
		self.term_term_ratio.fillna(0, inplace=True)

		console_log()
		console_log(self.doc_term_freq, '\n')
		console_log(self.term_term_freq, '\n')
		console_log(self.term_term_ratio, '\n')
		# model.to_csv('a.csv')

	def getTopWords(self):
		# geting the top words influencing topic
		for t in self.topic_word_distr.columns:
			console_log('{}\n{}'.format(t, '='*10))
			console_log(self.topic_word_distr[t].sort_values(ascending=False)[:10], end='\n\n')
		return

	def run_iteration(self):
		# the occurrence of words
		occurrence = (self.term_term_freq > 0).astype(np.int64)

		# the abundance
		abundance = occurrence.sum(0)
		
		# the mean abundance
		mean_abundance = abundance.mean()

		# the abundant words
		upper_percentile = abundance[abundance >= mean_abundance]
		upper_percentile = upper_percentile[upper_percentile <= upper_percentile.mean()]

		# the abundant words indices
		indices = upper_percentile.index

		# duplicate term freq and ratio
		term_term_freq = self.term_term_freq.copy()
		term_term_ratio = self.term_term_ratio.copy()

		# the most abundant words (they decide on topics)
		term_term_freq = term_term_freq[indices].T[indices].T
		term_term_ratio = term_term_ratio[indices].T[indices].T

		# adjust term ratio with trust factor
		term_term_ratio = self.trustFactor(term_term_freq) * term_term_ratio
		ttr = term_term_ratio * 0 #temp term term ratio matrix

		console_log('-'*50, 'Inferring word co-occurrence!', '-'*50)

		# infer word for word
		for w1 in tqdm(term_term_ratio.columns):
			for w2 in term_term_ratio:
				ttr[w1] += term_term_ratio[w1][w2] * term_term_ratio[w2]
		
		# normalize
		term_term_ratio = ttr / len(indices) if len(indices) > 0 else 0

		console_log(term_term_ratio, '\n')
		# input('enter to continue!')

		# create topic word distribution
		topic_word_distr = self.doc_term_freq.T[indices].T.apply(self.trustFactor)
		twd = topic_word_distr * 0 #temp topic word distribution matrix

		console_log('-'*50, 'Building Topic word distribution!', '-'*50)

		for topic_index in tqdm(topic_word_distr.columns):
			# topic word words
			twdd = topic_word_distr[topic_index]

			# inference based on word co-occurrence
			for word in indices:
				twd[topic_index] += twdd * term_term_ratio[word]

		# update the term doc freq
		topic_word_distr = twd / len(indices) if len(indices) > 0 else 0

		# clusters found
		topics = self.getClusters(topic_word_distr)

		new_topic_word_distr = {}
		for topic_index, doc_indices in enumerate(topics):
			new_topic_word_distr[topic_index] = topic_word_distr[doc_indices].mean(1)

		# update the topic distr
		self.topic_word_distr = pd.DataFrame(new_topic_word_distr)

		# console_log new model for iteration
		console_log(self.topic_word_distr, '\n')
		return

	def train(self, docs, n=1.0):
		# the documents that word appears in
		docs_length = len(docs)

		# the batch to learn
		docs_length = int(n * docs_length) if type(n) == float else n

		# ---------------------------------------------preprocessing----------------------------------
		console_log('-'*50, 'Preprocessing!', '-'*50, '\n')

		# constructs a model
		self.constructDTF(docs)

		# ----------------------------------------training----------------------------------------
		console_log('-'*50, 'Training!', '-'*50, '\n')

		# iterate and infer
		self.run_iteration()
		
		# set topic word_distr
		topic_word_distr = self.topic_word_distr.T

		# show the topic distribution
		console_log('{}{} Topic (word) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		console_log(topic_word_distr, end='\n\n')

		# geting the percentage influence of word to topic
		# self.topic_word_distr /= topic_word_distr.sum(1)
		
		
		# console_log('{}{} Topic (word(%)) distribution! {}{}'.format(' '*25, '='*25, '='*25, ' '*25))
		# console_log(self.topic_word_distr.T, end='\n\n')

		# self.topic_word_distr = topic_word_distr
		self.getTopWords()

		console_log('{} doc(s) read and {} word(s) in the vocabulary'.format(docs_length, len(self.word_docs)))
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
		words = map(lambda word: word.lower().replace("\n", '.'), word_tokenize(text))
		
		# remove stopwords
		words = [word for word in words if word not in cachedStopWords]
		
		# stemming
		# tokens =(list(map(lambda token: PorterStemmer().stem(token), words)));
		tokens = words

		# filter the lower than minimum length
		min_length = 3
		p = re.compile('[a-zA-Z]+')
		filtered_tokens = list(filter(lambda token: p.match(token) and len(token)>=min_length, tokens))
		# filtered_tokens = tokens

		return filtered_tokens

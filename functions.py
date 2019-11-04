# from std lib
import os, re

# from third party lib
import pandas as pd
import numpy as np

log_state = 1

if log_state == True:
	console_log = print

else:
	def console_log(*args):
		pass

"""Pagerank algorithm with explicit number of iterations.

Returns
-------
ranking of nodes (pages) in the adjacency matrix

"""


def pagerank(M, number_of_iterations=100, d=0.85):
	"""pagerank: The trillion dollar algorithm.

	Parameters
	----------
	M : numpy array
		adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
		sum(i, M_i,j) = 1]
	num_iterations : int, optional
		number of iterations, by default 100
	d : float, optional
		damping factor, by default 0.85

	Returns
	-------
	numpy array
		a vector of ranks such that v_i is the i-th rank from [0, 1],
		v sums to 1

	"""
	N = M.shape[1]
	v = np.ones((N, 1)) / N

	v_last = 0
	# while np.all(abs(v_last - v) > 10e-5):
	for _ in range(number_of_iterations):
		if np.all(abs(v_last - v) <= 10e-500):
			break

		v_last = v.copy()
		v = d * np.matmul(M.values, v) + (1 - d) / N

	return v

def read_txt(filename):
	with open(filename) as f:
		return f.read()

def getData(training_folder):
	filenames = os.listdir(training_folder)
	data = pd.DataFrame({'text':['' for _ in filenames]})

	for i, filename in enumerate(filenames):
		console_log('loading [{}]'.format(filename))
		data['text'][i] = read_txt('{}/{}'.format(training_folder, filename))

	console_log()
	return data

def getDFData(doc_path, column, n=-1):
	# load the corpus
	papers = pd.read_csv(doc_path)

	# Print head
	print(papers[column])
	
	'''
	iterate through the corpus to get the docs
	'''
	if type(n) == float:
		n = int(corpus.shape[0] * n)

	# Remove punctuation
	papers['text'] = papers[column].map(lambda x: re.sub('[,\.!?]', '', x))
	
	# Convert the titles to lowercase
	papers['text'] = papers['text'].map(lambda x: x.lower())

	return papers

def getNewsData(n=-1):
	doc_path='News_Category_Dataset_v2.json'

	# load the corpus
	corpus = pd.read_json(doc_path, lines=True)

	'''
	iterate through the corpus to get the docs
	'''
	if type(n) == float:
		n = int(corpus.shape[0] * n)

	for index, row in corpus.iterrows():
		if index == n:
			break
		yield row['short_description']

def perplexity(docs, nwt, ntd):
	nt = len(ntd)
	nd = np.sum(ntd, 0)
	n , ll = 0, 0.0
	for d, doc in enumerate(docs):
		for word in doc:
			ll += np.log(((nwt[word]/nt) * (ntd[:, d]/nd[d])).sum())
			n += 1
	return np.exp(ll/(-n))

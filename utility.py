# from std lib
import os, re, sys
from math import log as logarithm

# from third party lib
import pandas as pd
import numpy as np

log_state = 1

if log_state == True:
	console_log = print

else:
	def console_log(*args):
		pass

def log(num, base=10):
	if num > 0:
		return logarithm(num, base)

	return logarithm(sys.float_info.min * sys.float_info.epsilon, base)

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
			a = (nwt[word] / nt) if nt > 0 else 0 * nwt[word]
			b = (ntd[:, d] / nd[d]) if nd[d] > 0 else ntd[:, d] * 0
			ll += log((a * b).sum())
			n += 1
	return np.exp(ll/(-n) if n > 0 else 0)

# recursive_word_inference
Topic modeling for sparse data

## Overview
The allgorithm does topic modeling and it was developed to target sparse text and predict the topics accurately. The algorithm is not an implementation of any known algorithm or publish paper.

## How it works
- A document-word frequency matrix, document-topic, wor-word frequency, word-word ratio (relative to first word) matrix is constructed
- A word-topic matrix is created from doc-topic matrix based on most occuring ratio words
- A topic-word distibution is generated by combining the co-occurence of words and the word-topic matrix
- The topic-word distribution is used to infer topics for new documents
 
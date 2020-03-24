# Algorithm
The algorithm that describes the program is described below

## Terms
D - Number of documents in corpus
T - Number of topics in corpus
W - Number of words in corpus
wxwy - Number of co-occurenece of wx and wy
dwx - set of docs that wx exist in

## Train
1. Load data set (documents and topic labels)
    - preload document text
    - tokenize words
    - create vocabulary

2. create
    - document-topic matrix
        document-topic matrix = Matrix(D, T) #cdoc and related topic
        
    - document-word matrix
        document-word matrix = Matrix(D, W) #num of occurence of word in doc

    - words-in-document list
        #list of docs that word occurs in
        List of List: [[d1, d3, d9], [d5, d6, d8, d4], [d5, d1]]

    - word-word co-occurence freq matrix
        word-word co-occurence freq matrix = Matrix(W, W) #num of co-occurence of words
        ==> len(dw1 n dw2)

    - word-word co-occurence ratio matrix
        word-word co-occurence ratio matrix = Matrix(W, W) #ratio of word-word co-occurence freq with num of occurence of word
        ==> len(dw1 n dw2) / len(dw1)

3. get word-document freq matrix from document-word freq transpose
    ==> transpose(document-word freq)

4. derive word-topic matrix from word-document freq matrix
    - word-topic matrix = Matrix(W, T) #sum of doc-topic co-occurence for docs that word exist in

    - word-topic matrix = topic-word-distr = normalize(word-topic matrix)
    - remove word where any row of word-topic[word] <= (word-topic[word].mean() + word-topic.min())

5. get the most informative words
    - best-words-indices = topic-word[word-topic[topic] > word-topic[topic].mean()] # pick the works that are most informative i.e words that appear most for a specific topic compared to other words

6. infer term_term_ratio for best words columns
    ttr = copy(word-word-ratio)
    for w1 in best-words-indices
        for w2 in bast-words-indices
            factor = mean(topic-word-distr[w1] * topic-word-distr[w1])
            if factor <= 0: factor = 1
            ttr[w2] += factor * word-word-ratio[w1] 
    
    word-word-ratio[best-words-indices] = ttr / len(best-words-indices)

7. use best-words-indices to infer other words
    ttr = copy(word-word-ratio)
    for w1 in best-words-indices
        for w2 in word-word-ratio
            factor = mean(topic-word-distr[w1] * topic-word-distr[w1])
            if factor <= 0: factor = 1
            ttr[w2] += factor * word-word-ratio[w1] 
    
    word-word-ratio = ttr / len(best-words-indices)


## Test(doc classification)
1. load document
2. tokenize document text
    - tokens = tokenize(text)
3. create doc-word-distr
    for word in tokens
        if word in topic-word-distr
            words-in-doc.save(word)
            doc-word-distr += word-word-ratio[word]

    doc-word-disr = doc-word-distr / len(words-in-doc)

4. infer the relationship between doc and topics by identifying the words that exists in the test document, their relation to other words and inturn their relationship to topics: word-topic-distr for the document is extrapolated
    for word in words-in-doc
        topic-distr += doc-word-distr[word] * topic-word-distr[word]
    topic-distr = topic-distr / len(words-in-doc)

5. sort topic-distr to identify the topic that is most related to the test document
import re
import numpy as np
import pandas as pd
import collections
import os
import matplotlib.pyplot as plt
import csv


def get_all_words(sentences, wordsWithVec):
    wordsAll = set()
    i = 0
    for sentence in sentences:
        if i %1000 == 0:
            print(i)
        words = re.split("[-{}()_#@; |!,\*\n.:/?=]",str(sentence).lower())
        wordsAll = wordsAll | set(words)
        i += 1
    return list(wordsAll & set(wordsWithVec)) 


def writeNewWordVecMap(words, word_to_vec_map, outputFileName):
    file = open(outputFileName, 'w')
    for word in words:
        vs = word_to_vec_map[word]
        line = '{} {}\n'.format(word, ' '.join([str(v) for v in vs]))
        file.write(line)
    file.close()
    

def convertData2VecDF(dataDf):
    """
    This function will take a dataframe with the second colum being the 
    sentence, convert to the average of each word vector
    
    input: 
           dataframe with the second column being different setences.
    return: 
           dataframe with only one colume being the average of word vec.
    """
    val = []
    for i in range(dataDf.shape[0]):
        val.append(sentence_to_avg(test.ix[i, 1], word_to_vec_map, words))
        if i % 1000 == 0:
            print(i)
        i += 1
    return pd.DataFrame(val) 


def sentence_to_avg(sentence, word_to_vec_map, wordsAll):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the 
    GloVe representation of each word and averages its value into a single 
    vector encoding the meaning of the sentence.
    
    Arguments:
        sentence        -- string, one training example from X
        word_to_vec_map -- dictionary mapping every word in a vocabulary into 
                           its 50-dimensional vector representation
    
    Returns:
         avg            -- average vector encoding information about the 
                           sentence, numpy-array of shape (50,)
    """
    
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = re.split("[-{}()_#@; |!,\*\n.:/?=]",str(sentence).lower())
    
    # Initialize the average word vector, should have the same shape 
    # as your word vectors.
    DIM = 300   
    avg = np.zeros(DIM)
    cnt = 0

    # Step 2: average the word vectors. You can loop over the words in 
    # the list "words".
    for w in words:
        if w in wordsAll:
            avg += word_to_vec_map[w]
            cnt += 1
    if cnt != 0:
        avg = avg / cnt
        
    return avg


def findNullRows(df):
    return df.ix[pd.isnull(X_train).any(axis = 1)].index.tolist()


def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return word_to_vec_map


def read_glove_vecs_with_index(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices 
    corresponding to words in the sentences. The output shape should be such 
    that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
        X             -- array of sentences (strings), of shape (m, 1)
        word_to_index -- a dictionary containing the each word mapped to its 
                         index
        max_len       -- maximum number of words in a sentence. You can assume 
                         every sentence in X is no longer than this. 
    
    Returns:
        X_indices     -- array of indices corresponding to words in the 
                         sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]  # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape
    X_indices = np.zeros((m, max_len))
    allWords = word_to_index.keys()
    for i in range(m):  # loop over training examples  
        # Convert the ith sentence in lower case and split is into words.
        sentence_words = re.split("[-{}()_#@; |!,\*\n.:/?=]",str(X[i]).lower())
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if w in allWords:
                # Set the (i,j)th entry of X_indices to the index of the correct 
                # word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1
            if j >= max_len:
                break
    
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 
    50-dimensional vectors.
    
    Arguments:
        word_to_vec_map -- dictionary mapping words to their GloVe vector 
                           representation.
        word_to_index   -- dictionary mapping from words to their indices in the
                           vocabulary (400,001 words)

    Returns:
        embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1      # adding 1 to fit Keras embedding
    emb_dim = word_to_vec_map["cucumber"].shape[0]   # dim of GloVe word vectors
    
    # Initialize the embedding matrix as a numpy array of zeros of shape 
    # (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector 
    # representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make 
    # it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of 
    # the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. The layer
    # is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def tfidfTransform(texts, analyzer = 'word', ngram_range = (1, 1), 
                   max_features = None):
    """
    perform tf-idf conversion to the list of texts
    input: texts can be [train_text_list, test_text_list]
    output: converted textList in sparse matrix form, to view it use .todense()
    """
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True, #Apply sublinear tf scaling
        strip_accents='unicode',
        analyzer=analyzer,
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=ngram_range,
        max_features=max_features)
    
    word_vectorizer.fit(all_text)
    outputs = []
    for text in texts:
        outputs.append(word_vectorizer.transform(text))
    
    return outputs

import re
import numpy as np
import pandas as pd
import collections
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
        line = word + ' '+ ' '.join([str(v) for v in word_to_vec_map[word]]) + '\n'
        file.write(line)
    file.close()
    
def convertData2VecDF(dataDf):
    """
    This function will take a dataframe with the second colum being the sentence,
    convert to the average of each word vector
    input: dataframe with the second column being different setences.
    return: dataframe with only one colume being the average of word vec.
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
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = re.split("[-{}()_#@; |!,\*\n.:/?=]",str(sentence).lower())
    
    # Initialize the average word vector, should have the same shape as your word vectors.
    DIM = 300   
    avg = np.zeros(DIM)
    cnt = 0
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        if w in wordsAll:
            avg += word_to_vec_map[w]
            cnt += 1
    if cnt != 0:
        avg = avg / cnt
    
    ### END CODE HERE ###
    
    return avg

def findNullRows(df):
    return df.ix[pd.isnull(X_train).any(axis = 1)].index.tolist()

class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = valid_word_idx
        for i in range(vocab_size):
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
    

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

        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    





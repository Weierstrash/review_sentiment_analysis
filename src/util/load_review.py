import numpy as np
from tensorflow import keras

def return_tensored_review(file_name):
    #word index
    word_index = keras.datasets.imdb.get_word_index()
    dimension = 10000
    tensored_review = np.zeros(dimension)
    #opening the text file
    with open(file_name) as text_file:
        #reading each line
        for line in text_file:
            #reading each word
            for word in line.split():
                #checking to see if this is in the word index
                #and if its within the range
                if word in word_index.keys() and (word_index[word] <= 10000):
                    index = word_index[word]
                    tensored_review[index] = 1

    return tensored_review
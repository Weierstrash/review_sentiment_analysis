import numpy as np
from keras.datasets import imdb

#only keeping the top 10000 most frequently occuring words in the training data
(train_data, train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension = 10000):

    #creating an all zero array of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    #setting specific indices to 1s
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

#vectorizing both data sets
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#vectorizing both labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


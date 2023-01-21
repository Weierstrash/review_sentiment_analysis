from keras.datasets import imdb


def give_review(data):
    #a dictionary which maps words to integer indices
    word_index = imdb.get_word_index()

    #reversing the dictionary
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index]
    )

    #decoding the review using the reversed dictionary
    decoded_review = ' '.join(
        [reverse_word_index.get(i-3,'?') for i in data]
    )

    return decoded_review










import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data()
len(x_train)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

x_train[1]
decode_review(x_train[1])

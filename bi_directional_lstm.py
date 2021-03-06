# -*- coding: utf-8 -*-
"""Bi-Directional LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Safir-Mohammad-Mustak-Shaikh/Bi-Directional_LSTM_Sentiment_Classification_Keras/blob/master/Bi_Directional_LSTM.ipynb
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

max_features = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review

# Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")

# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 128)(inputs)

# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)

# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

#Load the IMDB movie review sentiment data
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

#Train and evaluate the model
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))

#New Predictions
from keras.datasets import imdb

new_review = ["""I absolutely adored this movie. For me, the best reason to see it is how stark a contrast it is from legal dramas like "Boston Legal" or "Ally McBeal" or even "LA Law." This is REALITY. The law is not BS, won in some closing argument or through some ridiculous defense you pull out of your butt, like the "Chewbacca defense" on South Park.) This is a real travesty of justice, the legal system gone horribly wrong, and the work by GOOD lawyers - not the shyster stereotype, who use all of their skills to right it. It will do more for restoring your faith in humanity than any Frank Capra movie or TO KILL A MOCKINGBIRD. And most importantly, I wept. During the film, during the featurette included at the end of the DVD - it's amazing. Wonderful film; wonderfully made. Thank God the filmmakers made it."""]
word_indices = imdb.get_word_index()
reviews = []
for doc in new_review:
  review = []
  for word in doc:
    if word not in word_indices:
      review.append(2)
    else:
      review.append(word_indices[word] + 3)
  review.sort(reverse=True)
  reviews.append(review)
x_test = keras.preprocessing.sequence.pad_sequences(reviews, truncating = 'post', padding = 'post', maxlen = maxlen)
print(model.predict(x_test))
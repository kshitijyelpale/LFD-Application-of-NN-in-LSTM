# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 02:00:59 2020

@author: Safir Mohammad
"""

#import libraries
import re
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
nltk.download('stopwords')
from nltk.corpus import stopwords


#define class
class SentimentModel:  
    
    def __init__(self):
        pass
    
    
    #Method for pre-processing data
    def preProcessData(self, x_test):
        
        print("Pre-processing data... Please wait!")
        
        #Fetch all stopwords and keep required stopwords
        all_stopwords = stopwords.words('english')
        my_stopwords = [ word for word in all_stopwords if word not in ("against", "up", "down", "out", "off", "over", "under", "more", "most", "each", "few", "some", "such", "no", "nor", "not", "only", "too", "very", "don", "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't")]
        
        #Get rid of special characters
        REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        
        for i in range(0,len(x_test)):
            #Keep only alphabets with single whitespace
            x_test[i] = REPLACE_NO_SPACE.sub("", x_test[i].lower())
            x_test[i] = REPLACE_WITH_SPACE.sub(" ", x_test[i])
            
            #Remove unwanted stopwords
            x_test[i] = x_test[i].split()
            x_test[i] = [ word for word in x_test[i] if word not in my_stopwords]
            x_test[i] = " ".join(x_test[i])
        
        print("Pre-processing done...!!!")
        return x_test
        
    
    #Method for Encoding Data
    def encodeData(self, x_test):
        
        print("Encoding data...")
        
        word_indices = imdb.get_word_index()
        reviews = []
        for doc in x_test:
            review = []
            for word in doc:
                if word not in word_indices:
                    review.append(2)
                else:
                    review.append(word_indices[word] + 3)
            review.sort(reverse=True)
            reviews.append(review)
            
        
        print("Encoding done...!!!")
        return reviews
    
    
    #Method for Embedding data
    def embeddData(self, max_doc_len, x_train, x_test):
        
        print("Embedding data...")
        
        #Word Embedding
        x_train = pad_sequences(x_train, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        x_test = pad_sequences(x_test, truncating = 'post', padding = 'post', maxlen = max_doc_len)
        
        print("Embedding done...!!!")
        return x_train, x_test
    
    
    #Method for creating ML->RNN->LSTM model
    def createModel(self, max_features, x_train, x_test, y_train, y_test):
        
        print('Build model...')
        model = Sequential()
        model.add(Embedding(max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        
        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        print('Train...')
        model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=6,
                  validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test,
                                    batch_size=32)
        print('Test score:', score)
        print('Test accuracy:', acc)
        
        return model
    
    
    #Method for predicting results through trained model
    def testModel(self, model, x_test):
        
        output = model.predict(x_test)
        print(output.shape)
        print(output)
        
    
def main():
            
        #Define vocabulary size
        max_features = 25000
        #Define number of words per document/review
        max_doc_len = 220
        
        sentimentModel = SentimentModel()
        
        print('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
        
        x_train, x_test = sentimentModel.embeddData(max_doc_len, x_train, x_test)
        
        #Build model
        model = sentimentModel.createModel(max_features, x_train, x_test, y_train, y_test)
             
        #Try with new review
        new_reviews = ["""I absolutely adored this movie. For me, the best reason to see it is how stark a contrast it is from legal dramas like "Boston Legal" or "Ally McBeal" or even "LA Law." This is REALITY. The law is not BS, won in some closing argument or through some ridiculous defense you pull out of your butt, like the "Chewbacca defense" on South Park.) This is a real travesty of justice, the legal system gone horribly wrong, and the work by GOOD lawyers - not the shyster stereotype, who use all of their skills to right it. It will do more for restoring your faith in humanity than any Frank Capra movie or TO KILL A MOCKINGBIRD. And most importantly, I wept. During the film, during the featurette included at the end of the DVD - it's amazing. Wonderful film; wonderfully made. Thank God the filmmakers made it."""]
        temp = []
        new_reviews = sentimentModel.preProcessData(new_reviews)
        new_reviews = sentimentModel.encodeData(new_reviews)
        temp, new_reviews = sentimentModel.embeddData(max_doc_len, temp, new_reviews)
        sentimentModel.testModel(model, new_reviews)

if __name__ == "__main__":
    main()

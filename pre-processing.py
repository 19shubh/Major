#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import ftfy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import numpy as np
import pandas as pd
import re

from math import exp
from numpy import sign
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import  classification_report, confusion_matrix, accuracy_score
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import PorterStemmer
from keras.preprocessing.sequence import pad_sequences

# In[2]:


#np.random.seed() is used to generate same set of numbers before rand() function is called
#random numbers work by starting with a number (the seed), multiplying it by a large number, 
#then taking modulo of that product. The resulting number is then used as the seed to generate the next "random" number.
#When you set the seed (every time), it does the same thing every time, giving you the same numbers.


# # Loading Data

# In[3]:


np.random.seed(1234)

# Both the sarcastic tweets and the random tweets are scrapped from twitter using twint
# In[4]:


sarcasm_df=pd.read_csv('sarcastic_tweets.csv',sep=',',header=None,usecols=range(1,2),nrows=10000)

#file should be in same path as ipynb file
#sep is by which fields are seperated.
#if the first row of the file can act as a header or not.
#if header=None then usecols is used to give column names.
#nrows is used to pick no of rows from file.


# In[5]:

random_df=pd.read_csv('random_tweets.csv',sep=',',header=None,usecols=range(1,8),nrows=12000)


# # Preprocessing of tweets

# 1. Removal of links, @, and hashtags and emojis from the tweets.
# 2. Corecting the encoding of the broken code using ftfy.
# 3. Expanding contracted text.
# 4. Removing of punctuations.
# 5. Removal of stopwords.
# 6. Stemming

# This all contractions are taken from the given link https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions

# In[6]:


clist= {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


# In[7]:


c_re = re.compile('(%s)' % '|'.join(clist.keys()))
#'|'.join(clist.keys()) is used to join all list members returned by clist.keys() 
#%s is string formatter.


# In[8]:


def expandContraction(text,c_re=c_re):
    def replace(match):
        return clist[match.group(0)]
    return c_re.sub(replace,text)
        


# In[9]:


def cleanTweets(tweets):
    c_t=[]      #array that will hold all tweets after cleaning and will be returned
    #working on each tweet.
    for tweet in tweets:
        tweet=str(tweet)  	
	#if the tweets doesnt contain URLs
 	if re.match("(\w+:\/\/\S+)", tweet) == None and len(tweet) > 10:
            #strings starting with https://
            #match function return a match object if the pattern is there in the stirng otherwise return None
            tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", tweet).split())
            #re.sub function is used to replace all occurences of a pattern in the given string
            #the property of hastags and tags that they are continuous after a @ or # sign is used to make RE.
        #tweet=ftfy.fix_text(tweet)#fixing faulty encoded text
        tweet = expandContraction(tweet)
        tweet = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", tweet).split())#removing punctuations
        #removing stop words
        s_w=set(stopwords.words('english'))
        w_t=nltk.tokenize.word_tokenize(tweet)#creates a list of all words in tweet
        fil_sen=[w for w in w_t if not w in s_w]
        tweet=' '.join(fil_sen)
        #stemming
        tweet=PorterStemmer().stem(tweet)
        c_t.append(tweet)
        
    return c_t


# In[53]:


sar_arr=[x for x in sarcasm_df[1]]
ran_arr=[x for x in random_df[7]]

fin_sar=cleanTweets(sar_arr)
fin_ran=cleanTweets(ran_arr)
print fin_sar
# In[ ]:





# # Tokenizing
# 

# Tokenizing is used to convert text into tokens. Each word is given a unique integer value.

# In[54]:


tokenizer=Tokenizer(20000)
tokenizer.fit_on_texts(fin_sar+fin_ran)


# In[55]:



#prints dictionary of words and their integer values using tokenizer.word_index

#Now sequence of words are being created.
# In[56]:


seq_sar=tokenizer.texts_to_sequences(fin_sar)
seq_ran=tokenizer.texts_to_sequences(fin_ran)

#text_to_word_sequence is used to create sequence of words.


# In[57]:



#a list of sublists is created. Each tweet is a sublist. instead of words thier integer values are taken....(seq_sar)


# In[58]:


#print(len(tokenizer.word_index))
#no of unique words


# In[59]:


data_d=pad_sequences(seq_sar,maxlen=140)
data_r=pad_sequences(seq_ran,maxlen=140)
#pad_sequence is used to make every sequence of same length)


# In[60]:


print(data_d.shape)


# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:45:25 2019

@author: Akshay
"""
import re
import string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

# In[57]:


def lemmatizeWords(text):
    wn = WordNetLemmatizer()
    #sentence = sent_tokenize(text)
    words=word_tokenize(text)
    listLemma = [wn.lemmatize(w,'v') for w in words]
    text=' '.join(listLemma)
    return text


# In[58]:


def stopWordsRemove(text):
    stopWordList = set(stopwords.words('english'))
    wordList=[x.lower().strip() for x in word_tokenize(text)]
    removedList=[x for x in wordList if not x in stopWordList]
    text=' '.join(removedList)
    return text


# In[59]:


def PreProcessing(text):
    text=lemmatizeWords(text)
    text=stopWordsRemove(text)
    return text

def PreProcessing_no_stop(text):
    text=lemmatizeWords(text)
    wordList=[x.lower().strip() for x in word_tokenize(text)]
    return ' '.join(wordList)


# In[59a]:


def PreProcessing_no_lemma(text):
    #text=lemmatizeWords(text)
    text=stopWordsRemove(text)
    return text


# In[60]:

def clean_text(text):
    text = re.sub(r"\n", "", text)    
    text = re.sub(r"\r", "", text) 
    text = re.sub(r"[0-9]", "digit", text)   
    text = re.sub(r"what’s", "what is ", text)
    text = re.sub(r"\’s", " ", text)
    text = re.sub(r"\’ve", " have ", text)
    text = re.sub(r"hasn’t", "has not ", text)
    text = re.sub(r"can’t", "cannot ", text)
    text = re.sub(r"n’t", " not ", text)
    text = re.sub(r"i’m", "i am ", text)
    text = re.sub(r"\’re", " are ", text)
    text = re.sub(r"\’d", " would ", text)
    text = re.sub(r"\’ll", " will ", text)
    text = re.sub(r"\’scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('&lt;','', text)
    text = re.sub('&gt;','', text)
    text = re.sub('&amp;','', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r' s ', ' ', text)
    text = text.strip(' ')
    return text

#%%

def remove_punctuations(input_string):
    '''Removes punctuations given a string'''
    output_string = input_string.translate(str.maketrans({key: None for key in string.punctuation}))
    return output_string

#%%

def remove_digit(text):
    return re.sub(r"digit", " ", text)   


#%%
def remove_spaces(text):
    return " ".join(text.split())
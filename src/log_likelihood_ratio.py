#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:02:44 2019

@author: sachitnagpal
"""
#%%
import nltk
import numpy as np
from collections import Counter
from scipy.stats import power_divergence
from cleaning import PreProcessing, clean_text, remove_punctuations

from nltk import sent_tokenize

#%%
similarity_parameter = 100
#%%
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def split_by_sentence(text):
    return tokenizer.tokenize(text)

#%%
def cleanText(text):
    text = PreProcessing(text)
    text = clean_text(text)
    text = remove_punctuations(text)
    return text

#%%
def LLR_all_words(input_text, corpus):
    global wcI, wcB
#    bg_corpus = [i for i in corpus if i!=input_text]
#    corpusTokens = nltk.word_tokenize(cleanText(' '.join(bg_corpus)))
    inputTokens = nltk.word_tokenize(cleanText(input_text))
    wcI = Counter(inputTokens)
#    wcB = Counter(corpusTokens)
    n1 = len(inputTokens)
    n2 = len(corpusTokens) - n1
    n=n1+n2
    LLR_words = {}
    for word in wcI:
        c1 = wcI[word]
        c2 = wcB[word] - c1
        p=(c1+c2)/n
        obs = np.array([[c1,n1],[c2,n2]])
        exp = np.array([[n1*p,n1],[n2*p,n2]])
        LLR_words[word] = power_divergence(f_exp=exp, f_obs=obs, lambda_=0).statistic[0]
    return LLR_words

#%%

from nltk import word_tokenize

def LLR_all_sentences(input_text, corpus, query, i):
    cleanQuery = cleanText(query)
    input_sentences = sent_tokenize(input_text)
    input_sentences = [sent for sent in input_sentences if len(word_tokenize(sent))>6]
    

    llrWords = LLR_all_words(input_text, corpus)
    output = []
    for sentence in input_sentences:
        d = {'sentence':sentence}
        cleanSent = cleanText(sentence)
        words = nltk.word_tokenize(cleanSent)
        d['scores'] = [llrWords.get(word, 0) if word not in query else max(similarity_parameter,llrWords.get(word, 0)) for word in words]
        d['score'] = np.mean([llrWords.get(word, 0) if word not in query else max(similarity_parameter,llrWords.get(word, 0)) for word in words])
        d['query_words'] = [word for word in words if word in nltk.word_tokenize(cleanQuery)]
        d['>10'] = [word for word in words if llrWords.get(word,0)>=10]
        d['_i'] = i
        output.append(d)
    return sorted(output, key=lambda x: x['score'], reverse=True)


def LLR_full_corpus(corpus, query, storyQuery):
    global corpusTokens, wcB
    out = []
    corpusTokens = nltk.word_tokenize(cleanText(' '.join(corpus)))
    wcB = Counter(corpusTokens)
    for input_text in enumerate(corpus):
        print(input_text[0])
        out.extend(LLR_all_sentences(input_text[1], corpus, query, input_text[0]))
    return sorted(out, key=lambda x: x['score'], reverse=True)

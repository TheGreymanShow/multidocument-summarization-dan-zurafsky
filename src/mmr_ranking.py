# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:24:15 2018

@author: Akshay
"""
from src.config import *
import numpy as np
import pandas as pd
from unidecode import unidecode
import requests
import re 
import nltk


def article_clean_up(text_):

    text = text_ if text_ is not None else ''

    regex = r'(?:(http|https|ftp|ftps)?(\:\/\/)?(www)?([a-zA-Z0-9\-\.]+\.)' \
            r'(com|org|edu|co|cms|in|uk|en|news|net|info|gov)(\/\S*)?)'

    text = re.sub(r'(\[\s*[^\]]*\s*\])', '', unidecode(text))

    text = re.sub(r'([a-z0-9]|\]|\)|\}|"|\')(\.\s*|\.\n*)([A-Z]|\{|\[|\(|")', r'\1. \3', text)
    text = re.sub(regex, '', text)

    return text


def article_pre_process(articles):

    bodies = []
    for article in articles:
        body = article_clean_up(article)
        bodies.append(body)

    return bodies


def first_iteration(sim_matrix):
    S = list()
    doc_query_sim = [row[-1] for row in sim_matrix]  #last column is query
    best_doc = np.argmax(np.array(doc_query_sim))
    S.append(best_doc)
    return S
    
def second_iteration(S, sim_matrix):
    X = S[0]
    doc_query_sim = [row[-1] for row in sim_matrix]  #last column is query
    doc_doc_sim = [sim_matrix[i][X] if i<X else sim_matrix[X][i] for i in range(len(sim_matrix))]
    MMR = [(1-diversity_param) * doc_query_sim[i] - diversity_param * doc_doc_sim[i] if i not in S else -100000 for i in range(len(sim_matrix)-1)]
    best_mmr = np.argmax(np.array(MMR))
    S.append(best_mmr)
    return S

def mmr_ranking(sim_matrix, n):
    S = list()
    S = first_iteration(sim_matrix)
    S = second_iteration(S, sim_matrix)
    doc_query_sim = [row[-1] for row in sim_matrix]  #last column is query
    
    while len(S) != summary_length:         # Third iteration until N
        doc_doc_sim = [[sim_matrix[i][X] if i<X else sim_matrix[X][i] for i in range(len(sim_matrix))] for X in S]
        best_doc_doc_sim = [doc_doc_sim[np.argmax(np.array([doc_doc_sim[i][j] for i in range(len(S))]))][j] for j in range(len(sim_matrix))]
        MMR = [(1-diversity_param) * doc_query_sim[i] - diversity_param * best_doc_doc_sim[i] if i not in S else -100000 for i in range(len(sim_matrix)-1)]
        best_mmr = np.argmax(np.array(MMR))
        S.append(best_mmr)

    return S


def rank_by_mmr(sim_matrix, n):    
    mmr_ranked_index = mmr_ranking(sim_matrix, n)
    return mmr_ranked_index

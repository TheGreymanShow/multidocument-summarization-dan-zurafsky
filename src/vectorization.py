from src.config import *
import gensim
import numpy as np
import pickle

UNFOUND_WORD2VEC_PATH = '5_million_sentences'
BIGRAMMER_PATH = '5_million_bigrammer.pickle.pickle'
TRIGRAMMER_PATH = '5_million_trigrammer.pickle'


def load_word2vec_model(path=UNFOUND_WORD2VEC_PATH):
    model = gensim.models.Word2Vec.load(path)
    return model


def load_phraser_models():
    with open(BIGRAMMER_PATH, 'rb') as file:
        bigrammer = pickle.load(file)
        file.close()

    with open(TRIGRAMMER_PATH, 'rb') as file:
        trigrammer = pickle.load(file)
        file.close()

    return bigrammer, trigrammer


def get_aggregate_vector(vectors):
    aggregate_vector = np.zeros((300, 1))
    for vec in vectors:
        vec = np.array(vec).reshape(300, 1)
        aggregate_vector += vec

    aggregate_vector = aggregate_vector/len(vectors)
    return aggregate_vector


def get_vector(word, model):
    """
    Function to fetch the vector for a word the model of pre-trained vectors.
    """
    return model.wv[word]

def vectorize_sentence(sentence, model):
    """
    Function to aggregate all the word vectors to form a sentence vector
    :param sentence: List of word vectors
    :param model: ConoceptNet Embedding model
    :return: A aggregated sentence vector
    """
    final_vec = np.zeros(300,)
    count = 0
    for word in sentence:
        count += 1
        dummy_vec = np.zeros(300,)
        try:
            temp_vec = get_vector(word, model)
            final_vec += temp_vec
        except:
            final_vec += dummy_vec
    return final_vec/count


def vectorize_sentence_weighted(sentence, model, weights):
    """
    Function to aggregate all the word vectors to form a sentence vector
    :param sentence: List of word vectors
    :param model: ConoceptNet Embedding model
    :return: A aggregated sentence vector
    """
    final_vec = np.zeros((300,))
    count = 0
    for i, word in enumerate(sentence):
        count += 1
        dummy_vec = np.zeros((300,))
        try:
            temp_vec = get_vector(word, model)
            final_vec += weights[i]*temp_vec
        except:
            final_vec += dummy_vec
    return final_vec/count

def word_overlap_similarity(sentence1, sentence2):
    """
    :param sentences: List of sentences
    :param keywords: Target sentence
    :return: List of scores, each score is similarity between the sentences
    """
    sentence1 = set(sentence1)
    sentence2 = set(sentence2)
    if len(sentence1.union(sentence2)) != 0:
        # return len((sentence1.intersection(sentence2))) / len(sentence1.union(sentence2))
        return len(sentence1.intersection(sentence2))
    else:
        return default_score

def word_overlap_similarity_weighted(sentence1, sentence2, weights):
    """
    :param sentences: List of sentences
    :param keywords: Target sentence
    :return: List of scores, each score is similarity between the sentences
    """
    if len(sentence1+sentence2) != 0:
        # return len((sentence1.intersection(sentence2))) / len(sentence1.union(sentence2))
        return np.sum([weights[i] for i,word in enumerate(sentence1) if word in sentence2])
    else:
        return default_score


def vectorize_documents(documents, model):
    """
    Function to aggregate all sentence vectors to form a document vector
    :param documents: document string or textssssss
    :param model: ConceptNet model of pre-trained vectors
    :return:
    """
    document_vectors = []
    count=0
    for document in documents:
        count+=1
        sentence_vectors = [vectorize_sentence(sentence, model) for sentence in document]
        document_vector = get_aggregate_vector(sentence_vectors)
        document_vectors.append(document_vector)
    return document_vectors


def vectorize_titles(titles, model):
    """
    Function to aggregate all titles vectors to form a document vector
    :param documents: document string or text
    :param model: ConceptNet model of pre-trained vectors
    :return:
    """
    sentence_vectors = [vectorize_sentence(sentence, model) for sentence in titles]
    return sentence_vectors

def compute_cosine_sim(vec1, vec2):
    numer = np.dot(vec1.reshape((300,)), vec2.reshape((300,)))
    denom = np.sqrt(np.sum(np.square(vec1.reshape(300, )))) * np.sqrt(
        np.sum(np.square(vec2.reshape(300, ))))
    
    similarity = numer/denom
    
    return similarity

#can add below fn to vectorization file
def vectorize_sentences(sentences, model):
    sentence_vectors = [vectorize_sentence(sent, model) for sent in sentences]
    return sentence_vectors
        

def transform_sentences(sentences, bigrammer, trigrammer):
    """Info"""
    bigrammed = [bigrammer[sent] for sent in sentences]
    trigrammed = [trigrammer[sent] for sent in bigrammed]
    return trigrammed
    
def transform_sentence(sent, bigrammer, trigrammer):
    """Info"""
    bigrammed = bigrammer[sent]
    trigrammed = trigrammer[bigrammed]
    return trigrammed
    

def normalize(scores, default_score):
    if len(scores) > 0:
        max_scores = max(scores)
        min_scores = min(scores)

        if max_scores == min_scores:
            return [default_score]*len(scores)
        scores = [float((score - min_scores) / (max_scores - min_scores)) for score in scores]
        return scores
    else:
        return [default_score]

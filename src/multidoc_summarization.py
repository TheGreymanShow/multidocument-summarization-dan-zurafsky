# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:28:21 2019

@author: Akshay
"""
from unidecode import unidecode
from src.cleaning import *
from src.vectorization import *
from src.mmr_ranking import *
from src.log_likelihood_ratio import *


BIGRAMMER_PATH = 'models/bigrammer_model.pickle'
TRIGRAMMER_PATH = 'models/trigrammer_model.pickle'
WORD2VEC_PATH = 'models/word2vec_model'
LAMBDA = 0.7

stop_words = set(stopwords.words('english'))


def clean(text):
    regex_1 = r"\.(?=\S)"
    regex_2 = r'(?:(http|https|ftp|ftps)?(\:\/\/)?(www)?([a-zA-Z0-9\-\.]+\.)' \
              r'(com|org|edu|co|cms|in|uk|en|news|net|info|gov)(\/\S*)?)'

    result = re.sub(regex_1, ". ", text)
    text = re.sub(r'(\[\s*[^\]]*\s*\])', '', unidecode(text))
    text = re.sub(r'([a-z0-9]|\]|\)|\}|"|\')(\.\s*|\.\n*)([A-Z]|\{|\[|\(|")', r'\1. \3', text)
    text = re.sub(regex_2, '', text)

    return text


# print(clean("Hello Akshay.I have come from america"))

def simplify(sentence):
    sentence = remove_punctuations(sentence)
    tokenized_sentence = word_tokenize(sentence)
    non_stopwords_sentence = [word for word in tokenized_sentence if word.lower() not in stop_words]
    return non_stopwords_sentence


def simplify_query(sentence):
    sentence = remove_punctuations(sentence)
    tokenized_sentence = word_tokenize(sentence)
    return tokenized_sentence


def load_word2vec_model(path=WORD2VEC_PATH):
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


def get_aggregate(a, b):
    return LAMBDA * a + (1 - LAMBDA) * b


def create_similarity_matrix(vector, doc_query_similarity):
    n = len(vector)

    matrix = [[compute_cosine_sim(vector[i], vector[j]) if j > i else 0 for j in range(n)] for i in range(n)]

    for i in range(len(matrix)):
        matrix[i].append(doc_query_similarity[i])  # last column is aggregate doc-query similarity
    matrix.append([0] * len(matrix[0]))  # add a dummy row to balance the newly added column

    return matrix


def multi_doc_summarize(documents, query):
    # 1. Preprocess raw data
    sources = [doc['publisher'] for doc in documents]
    documents = [doc['title'] + ". " + doc["body"] for doc in documents]
    clean_documents = [clean(doc) for doc in documents]
    split_documents = [sent_tokenize(doc) for doc in clean_documents]
    documents = [[sent for sent in doc if len(sent) > 7] for doc in split_documents]
    print("{} Input text sentences pre-processed".format(len(documents)))

    # 2. Merge all sentences
    all_sentences = [sentence for doc in documents for sentence in doc]  # stacked List comprehension
    sentence_sources = [sources[i] for i in range(len(documents)) for _ in documents[i]]  # stacked List comprehension
    print("{} Sentences merged from different documents".format(len(all_sentences)))

    # 3. simplyfy and remove unwanted words/parts of sentences
    cleaned_sentences = [simplify(sentence) for sentence in all_sentences]
    # cleaned_sentences = [sentence for sentence in cleaned_sentences if len(sentence)>5]
    print("{} Sentences cleaned".format(len(cleaned_sentences)))

    # 4. Data Pre-Processing (vectorization and overlap)
    model = load_word2vec_model()
    birammer, trigrammer = load_phraser_models()

    # 4a. Sentences
    transformed_sentences = transform_sentences(cleaned_sentences, birammer, trigrammer)
    sentence_vectors = [vectorize_sentence(sent, model) for sent in transformed_sentences]
    sentence_vectors = [np.array([vector[0] for vector in vec]) for vec in sentence_vectors]

    overlap_sentences = [remove_punctuations(sent) for sent in all_sentences]
    overlap_sentences = [clean_text(sent) for sent in overlap_sentences]
    overlap_sentences = [PreProcessing(sent) for sent in overlap_sentences]
    overlap_sentences = [simplify(sent) for sent in overlap_sentences]
    print("Vectorization and Overlap sentence data formed")

    # 4b. Query
    simple_query = simplify_query(query)
    transformed_query = transform_sentence(simple_query, birammer, trigrammer)
    query_vector = vectorize_sentence(transformed_query, model)
    query_vector = np.array([vec[0] for vec in query_vector])

    overlap_query = remove_punctuations(query)
    overlap_query = clean_text(overlap_query)
    overlap_query = [word.lower() for word in word_tokenize(overlap_query) if word.lower() not in stop_words]

    cosine_scores = normalize([compute_cosine_sim(query_vector, sent_vector) for sent_vector in sentence_vectors], 0.01)
    overlap_scores = normalize(
        [word_overlap_similarity(overlap_query, overlap_sent) for overlap_sent in overlap_sentences], 0.01)
    print("Vectorization and Overlap query data formed")

    doc_query_similarity = [get_aggregate(cosine, overlap) for (cosine, overlap) in zip(cosine_scores, overlap_scores)]
    print("Doc-query similarity calculated")

    # 5. Filter out sentences with no overlap with query
    no_overlap_indices = [i for i in range(len(overlap_scores)) if
                          overlap_scores[i] > 0 and len(cleaned_sentences[i]) > 10]
    print("{} No Overlap sentences filtered".format(len(no_overlap_indices)))

    # 6. Slice out the relevant sentences from all scores and lists
    doc_query_similarity = np.array(doc_query_similarity)[no_overlap_indices].tolist()
    sentence_vectors2 = np.array(sentence_vectors)[no_overlap_indices]
    print("Relevant data extracted after removing filtered data")

    # 7. Create similarity matrix
    doc_doc_similarity_matrix = create_similarity_matrix(sentence_vectors2, doc_query_similarity)
    print("Similarity Matrix created.")

    # 8. Rank using MMR
    mmr_ranked_index = rank_by_mmr(doc_doc_similarity_matrix, len(sentence_vectors2))
    print("MMR ranking successfull")

    # 9. Re-order according to MMR rank
    filtered_sentences = np.array(all_sentences)[no_overlap_indices]
    filtered_sources = np.array(sentence_sources)[no_overlap_indices]
    ranked_sentences = filtered_sentences[mmr_ranked_index].tolist()
    ranked_sources = filtered_sources[mmr_ranked_index].tolist()
    print("{} Sentences reordered according to MMR".format(len(ranked_sentences)))

    # 10. return summary
    best_sentences = ranked_sentences[:4]
    best_sources = ranked_sources[:4]
    print("Top {} Sentences selected for summary".format(len(ranked_sentences)))
    summary = " ".join(best_sentences)
    print("Smmarization successfull")
    return summary, best_sources


# 1. load input documents i.e a list of document strings of your choice
documents = []

# 2. specify the query for which answer is required
query = ""

# 3. Call function
summary, sources = multi_doc_summarize(documents, query)

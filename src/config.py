LAMBDA = 0.7          # proportion of weight to be given to cosine similarity measure. 1-LAMBDA weight is given to the overlap similarity
diversity_param = 0.3 # diversification parameter. increase for more diversity, and reduce to increase similarity of sentences with query
root_weight = 4       # weight of root word
children_weight = 1   # weight of dependents of root word
modifier_weight = 0   # weight of other words/modifiers of root word
nsimilar = 10         # number of similar words to consider for overlap
discount_similar_word = 1  # discounting factor for similar words
pair_type = 1         # indicates if pairs of sentences are needed. pair_type=0 will not make sentence pairs and consider single sentences. pair_type=1 makes pairs as (1,2), (3,4) and so on. value of 2 makes pairs as (1,2), (2,3), (3,4) and so on
min_doc_length = 7    # filters out documents which have less than min_doc_length sentences
min_sent_length = 6   # filters out sentences with less than min_sent_length_words
summary_length = 4    # length of summary in terms of number of sentences
default_score = 0.01  # default similarity if similarity score/normalization cannot be computed from data
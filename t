from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from collections import Counter

def Tokenize(column, seq_len, representation='bow'):
    ## Create BoW or TF-IDF representation
    if representation == 'bow':
        vectorizer = CountVectorizer()
    elif representation == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid representation. Choose 'bow' or 'tfidf'.")

    # Fit the vectorizer on the text column and transform the text to BoW or TF-IDF representation
    text_representation = vectorizer.fit_transform(column)

    # Convert the representation to an array
    text_array = text_representation.toarray()

    # Tokenize the columns text using the vocabulary
    text_int = []
    for text in column:
        r = [word for word in text.split()]
        text_int.append(r)

    # Add padding to tokens
    features = np.zeros((len(text_int), seq_len), dtype=int)
    for i, review in enumerate(text_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return text_array, features

# Example usage:
# Assuming 'your_column' is the column you want to process and seq_len is the desired sequence length
# BoW representation
sorted_words_bow, features_bow = Tokenize(your_column, seq_len, representation='bow')

# TF-IDF representation
sorted_words_tfidf, features_tfidf = Tokenize(your_column, seq_len, representation='tfidf')

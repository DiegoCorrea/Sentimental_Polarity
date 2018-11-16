import string
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def __lem_tokens(tokens):
    lemmer = nltk.stem.WordNetLemmatizer()
    return [lemmer.lemmatize(token) for token in tokens]


def __lem_normalize(text):
    remove_punct_dict = dict(
        (ord(punct), None)
        for punct in string.punctuation
    )
    return __lem_tokens(
        nltk.word_tokenize(
            text.lower().translate(remove_punct_dict)
        )
    )


def tf_as_matrix(sentence_list):
    LemVectorizer = CountVectorizer(tokenizer=__lem_normalize, stop_words='english')
    LemVectorizer.fit_transform(sentence_list)
    # print(LemVectorizer.vocabulary_)
    # print(sorted(LemVectorizer.vocabulary_.items(), key=lambda kv: kv[1])[:50])
    words_couted = LemVectorizer.vocabulary_.items()
    tf_matrix = LemVectorizer.transform(sentence_list).toarray()
    # print(tf_matrix)
    return tf_matrix, words_couted


def model_vectorize(dataset_df):
    tf_matrix, words_couted = tf_as_matrix(sentence_list=dataset_df['stem_sentence'].tolist())
    data_df = pd.DataFrame(data=np.matrix(tf_matrix), columns=[a for a, v in words_couted])
    return data_df, dataset_df['polarity']

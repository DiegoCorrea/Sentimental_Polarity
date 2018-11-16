import string
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
    word_position = LemVectorizer.vocabulary_.items()
    tf_matrix = LemVectorizer.transform(sentence_list).toarray()
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    return tfidf_matrix.toarray(), sorted(word_position , key=lambda k: (k[1], k[0]))


def model_vectorize(dataset_df):
    tfidf_matrix, word_position = tf_as_matrix(sentence_list=dataset_df['stem_sentence'].tolist())
    data_df = pd.DataFrame(data=np.matrix(tfidf_matrix), columns=[a for a, v in word_position])
    return data_df, dataset_df['polarity']

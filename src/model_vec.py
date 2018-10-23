import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer


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


def find_similarity(text_list):
    tfidf_vec = TfidfVectorizer(
        tokenizer=__lem_normalize,
        stop_words='english',
        analyzer='word'
    )
    tfidf = tfidf_vec.fit_transform([str(txt) for txt in text_list])
    return (tfidf * tfidf.T).toarray()


def test_similarity(text_list):
    LemVectorizer = CountVectorizer(tokenizer=__lem_normalize, stop_words='english')
    LemVectorizer.fit_transform(text_list)
    print(LemVectorizer.vocabulary_)
    tf_matrix = LemVectorizer.transform(text_list).toarray()
    print(tf_matrix)
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    print(tfidfTran.idf_)
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    for t in tfidf_matrix.toarray():
        print(t)
    print("")
    print("*")
    print("")
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    for c in cos_similarity_matrix:
        print(c)

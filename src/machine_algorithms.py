import numpy as np
from sklearn import metrics
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def train_naive_bayes(x_train, y_train):
    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)
    return clf


def train_knn(x_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=3)
    neigh.fit(x_train, y_train)
    return neigh


def train_tree(x_train, y_train):
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    clf = clf.fit(x_train, y_train)
    return clf


def train_perceptron(x_train, y_train):
    clf = Perceptron(tol=1e-3, random_state=0, n_jobs=3)
    clf.fit(x_train, y_train)
    return clf


def split_data(data_df, polarity_class):
    return train_test_split(data_df, polarity_class, test_size=0.20)


def evaluate(y_pred, y_test):
    return dict({
        'MAE': metrics.mean_absolute_error(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred)
    })


def main(data_df, polarity_class):
    evaluate_results_as_dict = dict()
    # Divisão dos dados em treinamento e teste
    x_train, x_test, y_train, y_test = split_data(data_df, polarity_class)
    # Uso dos dados no treinamento e teste do Perceptron RNA, por fim avaliação dos resultados
    print("\tPerceptron")
    clf = train_perceptron(x_train, y_train)
    y_pred = clf.predict(x_test)
    evaluate_results_as_dict['Perceptron'] = evaluate(y_pred, y_test)
    # Uso dos dados no treinamento e teste da Árvore de Decisão, por fim avaliação dos resultados
    print("\tÁrvore de Decisão")
    clf = train_tree(x_train, y_train)
    y_pred = clf.predict(x_test)
    evaluate_results_as_dict['AD'] = evaluate(y_pred, y_test)
    # Uso dos dados no treinamento e teste do KNN, por fim avaliação dos resultados
    print("\tKNN")
    clf = train_knn(x_train, y_train)
    y_pred = clf.predict(x_test)
    evaluate_results_as_dict['KNN'] = evaluate(y_pred, y_test)
    # Uso dos dados no treinamento e teste do Naive Bayes, por fim avaliação dos resultados
    print("\tNaive Bayes")
    clf = train_naive_bayes(x_train, y_train)
    y_pred = clf.predict(x_test)
    evaluate_results_as_dict['NB'] = evaluate(y_pred, y_test)
    return evaluate_results_as_dict

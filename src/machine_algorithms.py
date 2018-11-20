import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sys_variables import THREADS_NUMBER, N_NEIGHBORS, TEST_SIZE, MAX_DEPTH, MIN_SAMPLES_LEAF


def train_naive_bayes(x_train, y_train):
    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)
    return clf


def train_knn(x_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=THREADS_NUMBER)
    neigh.fit(x_train, y_train)
    return neigh


def train_tree(x_train, y_train):
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=MAX_DEPTH,
                                 min_samples_leaf=MIN_SAMPLES_LEAF)
    clf = clf.fit(x_train, y_train)
    return clf


def train_perceptron(x_train, y_train):
    clf = Perceptron(tol=1e-3, random_state=0, n_jobs=THREADS_NUMBER)
    clf.fit(x_train, y_train)
    return clf


def split_data(data_df, polarity_class):
    return train_test_split(data_df, polarity_class, test_size=TEST_SIZE)


def evaluate(y_pred, y_test):
    return mean_absolute_error(y_test, y_pred), precision_score(y_test, y_pred)


def main(data_df, polarity_class, run, model):
    result_df = pd.DataFrame(data=[], columns=['round', 'model', 'algorithm', 'metric', 'value'])
    # Divisão dos dados em treinamento e teste
    x_train, x_test, y_train, y_test = split_data(data_df, polarity_class)
    # Uso dos dados no treinamento e teste do Perceptron RNA, por fim avaliação dos resultados
    print("\tPerceptron")
    clf = train_perceptron(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, model, 'Perceptron', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, 'Perceptron', 'mae', mae]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste da Árvore de Decisão, por fim avaliação dos resultados
    print("\tÁrvore de Decisão")
    clf = train_tree(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, model, 'AD', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, 'AD', 'mae', mae]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste do KNN, por fim avaliação dos resultados
    print("\tKNN")
    clf = train_knn(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, model, str(N_NEIGHBORS) + 'NN', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, str(N_NEIGHBORS) + 'NN', 'mae', mae]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste do Naive Bayes, por fim avaliação dos resultados
    print("\tNaive Bayes")
    clf = train_naive_bayes(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, model, 'NB', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, 'NB', 'mae', mae]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    return result_df

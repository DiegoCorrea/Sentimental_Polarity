import pandas as pd
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


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
                           pd.DataFrame(data=[[run, model, 'KNN', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, 'KNN', 'mae', mae]],
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

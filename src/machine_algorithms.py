import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sys_variables import THREADS_NUMBER, N_NEIGHBORS, TEST_SIZE, MAX_DEPTH, MIN_SAMPLES_LEAF


def train_naive_bayes(x_train, y_train):
    """
    Função de treino do classificador Naive Bayes que retorna uma instancia treinada
    :param x_train: Dados de treinamento
    :param y_train: Classes dos dados de treinamento
    :return: Classificador treinado
    """
    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)
    return clf


def train_knn(x_train, y_train):
    """
    Função de treino do classificador KNN que retorna uma instancia treinada
    As constantes estão no arquivo de variaveis de sistema
    :param x_train: Dados de treinamento
    :param y_train: Classes dos dados de treinamento
    :return: Classificador treinado
    """
    neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=THREADS_NUMBER)
    neigh.fit(x_train, y_train)
    return neigh


def train_tree(x_train, y_train):
    """
    Função de treino do classificador Decision Tree que retorna uma instancia treinada
    As constantes estão no arquivo de variaveis de sistema
    :param x_train: Dados de treinamento
    :param y_train: Classes dos dados de treinamento
    :return: Classificador treinado
    """
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=MAX_DEPTH,
                                 min_samples_leaf=MIN_SAMPLES_LEAF)
    clf = clf.fit(x_train, y_train)
    return clf


def train_perceptron(x_train, y_train):
    """
    Função de treino do classificador Perceptron que retorna uma instancia treinada
    As constantes estão no arquivo de variaveis de sistema
    :param x_train: Dados de treinamento
    :param y_train: Classes dos dados de treinamento
    :return: Classificador treinado
    """
    clf = Perceptron(tol=1e-3, random_state=0, n_jobs=THREADS_NUMBER)
    clf.fit(x_train, y_train)
    return clf


def split_data(data_df, polarity_class):
    """
    Função particionadora dos dados em treino e teste
    As constantes estão no arquivo de variaveis de sistema
    :param data_df: Dados atributo/valor
    :param polarity_class: Classe dos dados
    :return: Quatro valores: dados de treino, classe de treino, dados de teste, classe de teste
    """
    return train_test_split(data_df, polarity_class, test_size=TEST_SIZE)


def evaluate(y_pred, y_test):
    """
    Função que valida o treinamento com as métricas MAE e Precision
    :param y_pred: Entrada predita
    :param y_test: Entrada real
    :return: Um par: primeiro o valor do MAE, segundo o valor do Precision
    """
    return mean_absolute_error(y_test, y_pred), precision_score(y_test, y_pred)


def main(x_train, x_test, y_train, y_test, run, model):
    """
    Função principal que executa todos os algoritmos selecionados, retornando um dataframe com os resultados
    :param x_train: Dados de treinamento
    :param x_test: Classes dos dados de treinamento
    :param y_train: Dados de teste
    :param y_test: Classes dos dados de teste
    :param run: Rodada do sistema
    :param model: Nome do modelo de dados a ser processado
    :return: DataFrame com os resultados de cada algoritmo
    """
    result_df = pd.DataFrame(data=[], columns=['round', 'model', 'algorithm', 'metric', 'value'])
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
                           pd.DataFrame(data=[[run, model, 'AD-' + str(MAX_DEPTH), 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, 'AD-' + str(MAX_DEPTH), 'mae', mae]],
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
                           pd.DataFrame(data=[[run, model, str(N_NEIGHBORS) + '-NN', 'precision', precision]],
                                        columns=['round', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, model, str(N_NEIGHBORS) + '-NN', 'mae', mae]],
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

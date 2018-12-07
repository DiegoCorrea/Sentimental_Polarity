import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import precision_score, mean_absolute_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sys_variables import THREADS_NUMBER, N_NEIGHBORS_CONFIG_1, MAX_DEPTH_CONFIG_1, MIN_SAMPLES_LEAF_CONFIG_1


def train_naive_bayes(x_train, y_train):
    """
    Função de treino do classificador Naive Bayes que retorna uma instancia treinada
    :param x_train: Dados de treinamento
    :param y_train: Classes dos dados de treinamento
    :return: Classificador treinado
    """
    clf = MultinomialNB()
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
    neigh = KNeighborsClassifier(n_neighbors=N_NEIGHBORS_CONFIG_1, n_jobs=THREADS_NUMBER)
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
    clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=MAX_DEPTH_CONFIG_1,
                                 min_samples_leaf=MIN_SAMPLES_LEAF_CONFIG_1)
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
    result_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    # Uso dos dados no treinamento e teste do Perceptron RNA, por fim avaliação dos resultados
    print("\t\tPerceptron")
    clf = train_perceptron(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, 'config_1', model, 'Perceptron', 'precision', precision]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, 'config_1', model, 'Perceptron', 'mae', mae]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste da Árvore de Decisão, por fim avaliação dos resultados
    print("\t\tÁrvore de Decisão")
    clf = train_tree(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(
                               data=[[run, 'config_1', model, 'AD', 'precision', precision]],
                               columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, 'config_1', model, 'AD', 'mae', mae]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste do KNN, por fim avaliação dos resultados
    print("\t\tKNN")
    clf = train_knn(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[
                               [run, 'config_1', model, 'KNN', 'precision', precision]],
                               columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, 'config_1', model, 'KNN', 'mae', mae]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    # Uso dos dados no treinamento e teste do Naive Bayes, por fim avaliação dos resultados
    print("\t\tNaive Bayes")
    clf = train_naive_bayes(x_train, y_train)
    y_pred = clf.predict(x_test)
    mae, precision = evaluate(y_pred, y_test)
    result_df = pd.concat([result_df,
                           pd.DataFrame(data=[[run, 'config_1', model, 'NB', 'precision', precision]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value']),
                           pd.DataFrame(data=[[run, 'config_1', model, 'NB', 'mae', mae]],
                                        columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
                           ],
                          sort=False
                          )
    return result_df

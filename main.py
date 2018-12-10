import pandas as pd
from sklearn.model_selection import train_test_split

from src import generate_results
from src import machine_algorithms_config_1
from src import machine_algorithms_config_2
from src import pmi_model
from src import preprocessing
from src import tdidf_model
from src.mining_data import make_dataset
from sys_variables import EXECUTION_TIMES
from sys_variables import TEST_SIZE


def split_data(data_df, polarity_class):
    """
    Função particionadora dos dados em treino e teste
    As constantes estão no arquivo de variaveis de sistema
    :param data_df: Dados atributo/valor
    :param polarity_class: Classe dos dados
    :return: Quatro valores: dados de treino, classe de treino, dados de teste, classe de teste
    """
    return train_test_split(data_df, polarity_class, test_size=TEST_SIZE)


def split_by_index(index_list, polarity_class):
    return train_test_split(index_list, polarity_class, test_size=TEST_SIZE)


def split_tfidf(tfidf_pattern, x_train, x_test):
    return tfidf_pattern.ix[x_train], tfidf_pattern.ix[x_test]


def split_pmi(pmi_model, x_train, x_test):
    return pmi_model.ix[x_train], pmi_model.ix[x_test]


if __name__ == '__main__':
    # Load Dataset
    print("1.\tCriando o Dataset")
    DATASET = make_dataset()
    DATASET.info(memory_usage='deep')
    print("\n")
    print("2.\tPre Processamento")
    DATASET = preprocessing.main_start(DATASET)
    DATASET.info(memory_usage='deep')
    print("\n")
    print("3.\tPreparando os modelos")
    print("\t\tTF-IDF")
    tfidf_pattern = tdidf_model.mold(DATASET)
    print("\t\tPMI")
    pmi_model = pmi_model.mold(DATASET)
    print("\n")
    results_df = pd.DataFrame(data=[], columns=['round', 'config', 'model', 'algorithm', 'metric', 'value'])
    print("4.\tAprendizado")
    for i in range(EXECUTION_TIMES):
        print("- Rodada " + str(i + 1))
        # Divisão dos dados em treinamento e teste
        x_train, x_test, y_train, y_test = split_by_index(tfidf_pattern.index.tolist(), DATASET['polarity'])
        tfidf_x_train, tfidf_x_test = split_tfidf(tfidf_pattern, x_train, x_test)
        pmi_x_train, pmi_x_test = split_pmi(pmi_model, x_train, x_test)
        print(pmi_x_train)
        print('=' * 90)
        print(pmi_x_test)
        print("\tCONFIG 1 - TFIDF")
        results_df = pd.concat(
            [results_df,
             machine_algorithms_config_1.main(tfidf_x_train, tfidf_x_test, y_train, y_test, i + 1,
                                              'TFIDF')],
            sort=False)
        print("\tCONFIG 1 - PMI")
        results_df = pd.concat(
            [results_df,
             machine_algorithms_config_1.main(pmi_x_train, pmi_x_test, y_train, y_test, i + 1,
                                              'PMI')],
            sort=False)
        print("\tCONFIG 2 - TFIDF")
        results_df = pd.concat(
            [results_df,
             machine_algorithms_config_2.main(tfidf_x_train, tfidf_x_test, y_train, y_test, i + 1,
                                              'TFIDF')],
            sort=False)
        print("\tCONFIG 2 - PMI")
        results_df = pd.concat(
            [results_df,
             machine_algorithms_config_2.main(pmi_x_train, pmi_x_test, y_train, y_test, i + 1,
                                              'PMI')],
            sort=False)
    generate_results.graphics(results_df)
    generate_results.comparate(results_df)

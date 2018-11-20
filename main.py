import pandas as pd

from src import graphics
from src import machine_algorithms
from src import preprocessing
from src import tdidf_model
from src.mining_data import make_dataset
from sys_variables import EXECUTION_TIMES

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
    tfidf_pattern, polarity_class = tdidf_model.mold(DATASET)
    print("\n")
    results_df = pd.DataFrame(data=[], columns=['round', 'model', 'algorithm', 'metric', 'value'])
    for i in range(EXECUTION_TIMES):
        print("4.\tAprendizado")
        # Divisão dos dados em treinamento e teste
        x_train, x_test, y_train, y_test = machine_algorithms.split_data(tfidf_pattern, polarity_class)
        results_df = pd.concat(
            [results_df, machine_algorithms.main(x_train, x_test, y_train, y_test, i + 1, 'TFIDF')],
            sort=False)
    graphics.generate(results_df)

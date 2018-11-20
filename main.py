from src.mining_data import make_dataset
from src.model_vec import model_vectorize
from src import machine_algorithms
from src import preprocessing
import pandas as pd


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
    results_df = pd.DataFrame(data=[], columns=['round', 'model', 'algorithm', 'metric', 'value'])
    for i in range(5):
        print("3.\tPreparando os modelos")
        data_df, polarity_class = model_vectorize(DATASET)
        print("\n")
        print("4.\tAprendizado")
        results_df = pd.concat([results_df, machine_algorithms.main(data_df, polarity_class, i, 'Model_1')], sort=False)
    print(results_df)

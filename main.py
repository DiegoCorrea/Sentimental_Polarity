from src.mining_data import make_dataset
from src.model_vec import model_vectorize
from src import machine_algorithms
from src import preprocessing

# Load dataset
DATASET = make_dataset()
print(DATASET.head())
pp = preprocessing.main_start(DATASET)
print(pp.head())
data_df, polarity_class = model_vectorize(pp)
print(pp.head())
machine_algorithms.main(data_df, polarity_class)

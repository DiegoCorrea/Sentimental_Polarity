from src.mining_data import make_dataset
from src.model_vec import model_vectorize
from src import machine_algorithms

# Load dataset
DATASET = make_dataset()
data_df, polarity_class = model_vectorize(DATASET)
print(data_df.head())
machine_algorithms.main(data_df, polarity_class)

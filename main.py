from src.mining_data import make_dataset
from src.model_vec import model_vectorize
from src import tree
from src import knn

print("1- Mining Data")
print("2- Start program")
keyboard_input = 1

DATASET = make_dataset()
data_df, polarity_class = model_vectorize(DATASET)
print(data_df.head())
tree.main(data_df, polarity_class)
knn.main(data_df, polarity_class)
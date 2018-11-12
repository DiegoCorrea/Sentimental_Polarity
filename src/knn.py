from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np


def train(x_train, y_train):
    neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=3)
    neigh.fit(x_train, y_train)
    return neigh


def split_data(data_df, polarity_class):
    return train_test_split(data_df, polarity_class, test_size=0.20)


def evaluate(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    # df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


def main(data_df, polarity_class):
    x_train, x_test, y_train, y_test = split_data(data_df, polarity_class)
    neigh = train(x_train, y_train)
    y_pred = neigh.predict(x_test)
    evaluate(y_pred, y_test)

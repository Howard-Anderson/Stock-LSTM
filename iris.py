"""
                Iris Flower Classification


"""

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def read(filename:str):
    return pd.read_csv(filename)

def input_fn(features, labels, training = True, batch_size = 255):
    data = tf.data.Dataset.from_tensor_slices(dict(zip(labels,features)))
    if training:
        data = data.shuffle(1000).repeat()
    return data.batch(batch_size)

if __name__ == "__main__":
    print("\n\nIris Classification:\n\n")
    #dataset = read(input("\nEnter the Dataset File with extension[.csv]: "))
    dataset = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(dataset["data"],dataset["target"], random_state = 34)
    feature_cols = []
    for key in dataset["feature_names"]:
        feature_cols.append(tf.feature_column.numeric_column(key = key))
    print(f"\n\nFeature Columns:\n {feature_cols}")

    # Building the Model:
    classifier = tf.estimator.DNNClassifier(
                    feature_columns = feature_cols,
                    hidden_units = [30,10],
                    n_classes = 3
                )

    # Training:
    classifier.train(input_fn = lambda: input_fn(x_train, y_train, training = True), steps = 5000)

    # Evaluation:
    eval_res = classifier.evaluate(input_fn = lambda: input_fn(x_test, y_test, training = False))

    print(f"\n\nTest Set Accuracy: {eval_res}")



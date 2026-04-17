import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from kNNClassifier import tune_knn, plot_classifier_preformance
from random_forest import train_rfc
from dim_red import dimension_reduction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
from multilayer_preceptron import MultilayerPerception

train_data = np.load("./data/train_matrix.npy")
train_labels = np.load("./data/train_labels.npy")

test_data = np.load("./data/test_matrix.npy")
test_labels = np.load("./data/test_labels.npy")
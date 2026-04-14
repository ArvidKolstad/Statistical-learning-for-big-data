import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier


def random_forest_classifier():

    classifier = RandomForestClassifier()


def main():
    random_forest_classifier_settings = {
        "criterion": "gini",
        "max_depth": "None",
        "min_samples_split": None,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
    }

    random_forest_classifier()


if __name__ == "__main__":
    main()

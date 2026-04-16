import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Test data
train_images = np.array([[0,1,0,1], [0,0,0,0], [1,0,0,1], [0,1,0,1], [0,0,0,0], [1,0,0,1], [0,1,0,1], [0,0,0,0], [1,0,0,1], [0,1,0,1], [0,0,0,0], [1,0,0,1]])
train_labels = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1])

test_images = np.array([[0,0,0,1], [1,1,1,0], [0,1,0,1], [1,0,1,0]])
test_labels = [1, 0, 1, 0]
digits = [0,1]

def plot_classifier_preformance(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=digits)
    ConfusionMatrixDisplay(cm, display_labels=digits).plot(cmap='Greens')

def tune_knn(x_train, y_train, k_values, n_folds):
    cv_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=n_folds, scoring="accuracy")
        cv_scores.append(scores.mean())

    plt.figure()
    plt.plot(k_values, cv_scores)
    plt.xlabel('k')
    plt.ylabel('CV Accuracy')
    plt.grid()

    best_k = k_values[np.argmax(cv_scores)]
    print(f'Best k value: {best_k}')

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x_train, y_train)
    return best_knn

knn = tune_knn(train_images, train_labels, range(1, 6), 2)
plt.show()

plot_classifier_preformance(knn, test_images, test_labels)
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")

train_label = train['label']
train_data = train.drop('label', axis = 1)

test_label = test['label']
test_data = test.drop('label', axis = 1)

print(train_data.shape)

plt.imshow(train_data.iloc[500].to_numpy().reshape(28,28))
plt.axis('off')
plt.show()

pca = PCA(n_components = 0.95)
tsne = TSNE(n_components = 2, random_state = 42)

pca_tsne = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('tsne', TSNE(n_components=2, random_state=42))
])

train_reduced = pca_tsne.fit_transform(train_data)
print(train_reduced.shape)

plt.figure(figsize=(12,8))
plt.scatter(train_reduced[:,0], train_reduced[:,1], c=train_label, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
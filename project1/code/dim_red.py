import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline


train_label = np.load('data/train_labels.npy')
train_data = np.load('data/train_matrix.npy')

test_label = np.load('data/test_labels.npy')
test_data = np.load('data/test_matrix.npy')

print(train_data.shape)

image_idx = random.randint(0, train_data.shape[0])
print(image_idx)

plt.imshow(train_data[500].reshape(28,28))
plt.axis('off')
plt.show()

# pca = PCA(n_components = 0.95)
# tsne = TSNE(n_components = 3, random_state = 42)

pca_tsne = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('tsne', TSNE(n_components=3, random_state=42))
])

train_reduced = pca_tsne.fit_transform(train_data)
print(train_reduced.shape)

plt.figure(figsize=(12,8))
plt.scatter(train_reduced[:,0], train_reduced[:,1], c=train_label, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()
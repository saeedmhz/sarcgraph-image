import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the feature vectors and labels
feature_vectors = np.load('./dino_feature_vectors.npy')
labels = ... # Example: np.load('path_to_labels.npy')

# Perform PCA to reduce the dimensionality to 2D
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(feature_vectors)

# Plot the PCA results with red and blue dots based on labels
plt.figure(figsize=(8, 6))
for label, color in zip([0, 1], ['red', 'blue']):
    mask = (labels == label)
    plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                c=color, label=f'Label {label}', alpha=0.7, edgecolors='k', s=10)

plt.title('PCA of Feature Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('./features-pca.png', dpi=300, bbox_inches='tight')
plt.show()
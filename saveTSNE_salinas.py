import matplotlib
matplotlib.use('Agg')  # For headless environments (e.g., SSH)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from pytorch_tabnet.tab_model import TabNetClassifier

# Load saved model
clf = TabNetClassifier()
clf.load_model('trained_tabnet_model.zip')  # <-- Update this if saved differently

# Load preprocessed test data and labels
Xtest = np.load('Xtest.npy')  # Shape: (N, C*H*W), e.g., (45837, 6250)
ytest = np.load('ytest.npy')  # Shape: (N,)

# Ensure ytest is a NumPy array
ytest = np.array(ytest)

# Get embeddings from model
clf.network.eval()
X_tensor = torch.from_numpy(Xtest).float().to(next(clf.network.parameters()).device)

with torch.no_grad():
    output, _ = clf.network(X_tensor)

embeddings = output.detach().cpu().numpy()

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(embeddings)

# Custom colormap (match classification map)
custom_colors = [
    "#0000FF", "#FF6400", "#00FF86", "#964696",
    "#6496FF", "#3C5A72", "#FFFF7D", "#FF00FF",
    "#6400FF", "#01AAFF", "#00FF00", "#AFAF52",
    "#64BE38", "#8C432E", "#73FFAC", "#FFFF00"
]
cmap_custom = ListedColormap(custom_colors)

# Plot t-SNE
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=ytest,
    cmap=cmap_custom,
    alpha=0.7,
    s=10
)

# Custom colorbar for class labels
unique_labels = np.unique(ytest)
cbar = plt.colorbar(scatter, ticks=unique_labels)
cbar.set_ticklabels([str(int(i + 1)) for i in unique_labels])  # show 1 to 16

# Aesthetic cleanup
plt.xticks([])
plt.yticks([])
plt.box(False)

# Save figure
plt.tight_layout()
plt.savefig("tabnet_tsne_visualization_posttraining.png", dpi=300)
plt.close()
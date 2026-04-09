import scipy.io as sio
import scipy
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(0)
import os
#import wget
from pathlib import Path
import shutil
import gzip
from matplotlib import pyplot as plt
from pytorch_tabnet.pretraining import TabNetPretrainer


#dataind=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Indian_pines_corrected.mat')  #
#labelind=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Indian_pines_gt.mat')

##salinas
datasalilnas=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Salinas_corrected.mat')  #
labelsalinas=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Salinas_gt.mat')

# datapavu=dataupav['paviaU']
# labelpavu=labelupav['paviaU_gt']

datasalilnas=datasalilnas['salinas_corrected']
labelsalinas=labelsalinas['salinas_gt']

#X=dataind['indian_pines_corrected']
#y=labelind['indian_pines_gt']

# X = datapavu
# y = labelpavu

X = datasalilnas
y = labelsalinas

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#test_ratio = 0.96
windowSize = 25

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
	
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
	
#flipping for X(N,C,H,W)
def padWithZeros(X, margin=2):
  #  newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    newX = np.zeros((X.shape[0], X.shape[1] + 2 * margin, X.shape[2] + 2* margin))  #flip padding
    x_offset = margin
    y_offset = margin
   # newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    newX[: , x_offset:X.shape[1] + x_offset, y_offset:X.shape[2] + y_offset] = X  #flip padding
    return newX	

#flipped channel and window
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[1] * X.shape[2], X.shape[0], windowSize, windowSize))
    patchesLabels = np.zeros((X.shape[1] * X.shape[2]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[1] - margin):
        for c in range(margin, zeroPaddedX.shape[2] - margin):
            patch = zeroPaddedX[:,r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

#flipping for X(N,C,H,W)
Xo,pca = applyPCA(X,numComponents=10)
Xres = Xo.reshape(Xo.shape[0]*Xo.shape[1],10)
Xres.shape
Xn=Xres.transpose()
Xn.shape
Xresn = Xn.reshape(10,Xo.shape[0],Xo.shape[1])
Xresn.shape

# #flipping for X(N,C,H,W), univ of pavia
# Xo,pca = applyPCA(X,numComponents=7)
# Xres = Xo.reshape(Xo.shape[0]*Xo.shape[1],7)
# Xres.shape
# Xn=Xres.transpose()
# Xn.shape
# Xresn = Xn.reshape(7,Xo.shape[0],Xo.shape[1])
# Xresn.shape	

#flipping padding for X(N,C,H,W) instead of X(N,H,W,C)
X, y = createImageCubes(Xresn, y, windowSize=25)
#X, y = createImageCubes(Xresn, y, windowSize=15)
X.shape, y.shape

import numpy as np
from sklearn.model_selection import train_test_split

import time

start_time = time.time()

num_classes = 16
samples_per_class = 200

X_selected = []
y_selected = []
selected_indices = []

# 1. Select 200 samples per class for training
for cls in range(num_classes):
    idx = np.where(y == cls)[0]
    if len(idx) < samples_per_class:
        print(f"Warning: class {cls} has only {len(idx)} samples.")
        chosen = np.random.choice(idx, len(idx), replace=False)
    else:
        chosen = np.random.choice(idx, samples_per_class, replace=False)

    X_selected.append(X[chosen])
    y_selected.append(y[chosen])
    selected_indices.extend(chosen)

X_selected = np.concatenate(X_selected, axis=0)
y_selected = np.concatenate(y_selected, axis=0)
selected_indices = np.array(selected_indices)
Xtrain = X_selected
ytrain = y_selected

print("Training set shape :", X_selected.shape, y_selected.shape)

# 2. Get remaining indices for test/validation
all_indices = np.arange(len(y))
remaining_indices = np.setdiff1d(all_indices, selected_indices)

X_remaining = X[remaining_indices]
y_remaining = y[remaining_indices]

## 3. Split remaining into validation and test (e.g., 20% val, 80% test)
#X_valid, X_test, y_valid, y_test = train_test_split(
#    X_remaining, y_remaining, test_size=0.8, stratify=y_remaining, random_state=42
#)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_remaining, y_remaining, test_size=0.9, stratify=y_remaining, random_state=42
)
print("Validation shape   :", X_valid.shape, y_valid.shape)
print("Test shape         :", X_test.shape, y_test.shape)

Xvalid = X_valid
yvalid = y_valid
Xtest = X_test
ytest = y_test


clf = TabNetClassifier(
    n_d=256, n_a=256, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=[],
    cat_dims=[],
    cat_emb_dim=[],
    lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    #optimizer_params=dict(lr=1e-3),
    scheduler_params = {"gamma": 0.95,
                     "step_size": 20},
    scheduler_fn=torch.optim.lr_scheduler.StepLR, epsilon=1e-15
)

Xtrain = Xtrain.reshape(3200,10*25*25)
##20% valid 80% test
#Xtest = Xtest.reshape(40744,10*25*25)
#Xvalid = Xvalid.reshape(10185,10*25*25)  
##10% valid 90% test
Xtest = Xtest.reshape(45837,10*25*25)
Xvalid = Xvalid.reshape(5092,10*25*25)

clf.fit(
    X_train=Xtrain, y_train=ytrain,
    eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],
    eval_name=['train', 'valid'],
    max_epochs=500, patience=100,     #max_epochs = 500, 1500, 100
    batch_size=64, virtual_batch_size=128
)

end_time = time.time()
total_time = end_time - start_time

print(f"Training completed in {total_time:.2f} seconds.")

preds = clf.predict(Xtest)
testacc=accuracy_score(y_pred=preds,y_true=ytest)
print('accuracy:',testacc)

y_pred = clf.predict(Xtest)
import sklearn.metrics as metrics
conf_matrix = metrics.confusion_matrix(y_pred, ytest)

each_class_accuracy_=conf_matrix.diagonal()/conf_matrix.sum(axis=1)
print('each class accuracy:',each_class_accuracy_)

average_accuracy=np.mean(each_class_accuracy_)
print('average accuracy:',average_accuracy)



end_time2 = time.time()
total_time_test = end_time2 - end_time
print(f"Training completed in {total_time_test:.2f} seconds.")
from sklearn.metrics import cohen_kappa_score,accuracy_score
kappa=cohen_kappa_score(ytest,y_pred)
print('kappa coefficient:',kappa)

#clf.save_model('trained_tabnet_model')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
import numpy as np
from matplotlib.colors import ListedColormap

np.random.seed(42)

# Ensure NumPy array
ytest = np.array(ytest)

# Step 1: Get embeddings
clf.network.eval()
X_tensor = torch.from_numpy(Xtest).float().to(next(clf.network.parameters()).device)

with torch.no_grad():
    output, _ = clf.network(X_tensor)

embeddings = output.detach().cpu().numpy()

# Step 2: t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(embeddings)

# Step 3: Define bold 16-color palette (ColorBrewer-inspired)
custom_colors = [
    "#0000FF", "#FF6400", "#00FF86", "#964696",
    "#6496FF", "#3C5A72", "#FFFF7D", "#FF00FF",
    "#6400FF", "#01AAFF", "#00FF00", "#AFAF52",
    "#64BE38", "#8C432E", "#73FFAC", "#FFFF00"
]
cmap_custom = ListedColormap(custom_colors)

# Step 3: Plot
plt.figure(figsize=(8, 6))

# Scatter plot
scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=ytest,
    cmap=cmap_custom,
    alpha=0.7,
    s=10
)

# Check class range
unique_labels = np.unique(ytest)
num_classes = len(unique_labels)

# Set colorbar ticks to match actual classes
cbar = plt.colorbar(scatter, ticks=unique_labels)
##cbar.set_label('Class Label')
cbar.set_ticklabels([str(int(i + 1)) for i in unique_labels])  #show 1 to 16

# Clean up
plt.xticks([])
plt.yticks([])
plt.box(False)
##plt.title("t-SNE Visualization of TabNet Embeddings", fontsize=12)

# Save to PNG
plt.tight_layout()
plt.savefig("tabnet_tsne_visualization_salinas_customcolornew.png", dpi=300)
plt.close()



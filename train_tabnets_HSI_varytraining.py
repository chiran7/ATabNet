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


dataind=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Indian_pines_corrected.mat')  #
labelind=scipy.io.loadmat('../TabNets/pytorch_tabnet/data/Indian_pines_gt.mat')

# datapavu=dataupav['paviaU']
# labelpavu=labelupav['paviaU_gt']

#datasalilnas=datasalilnas['salinas_corrected']
#labelsalinas=labelsalinas['salinas_gt']

X=dataind['indian_pines_corrected']
y=labelind['indian_pines_gt']

# X = datapavu
# y = labelpavu

#X = datasalilnas
#y = labelsalinas

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

#test_ratio = 0.93
test_ratio = 0.95
windowSize = 25

#def splitTrainTestSet(X, y, testRatio, randomState=345):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
#                                                        stratify=y)
#    return X_train, X_test, y_train, y_test

import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def splitTrainTestSet(X, y, testRatio=0.1, randomState=345):
    np.random.seed(randomState)
    
    # Dictionary to hold class-wise indices
    class_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_indices[label].append(idx)
    
    test_indices = []
    train_indices = []

    for label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)
        n_test = int(len(indices) * testRatio)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        test_indices.extend(test_idx)
        train_indices.extend(train_idx)

    # Create train/test sets
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

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

#Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
# Select 10% from each class for test
#Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, testRatio=0.9)

Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, testRatio=0.925)
#Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, testRatio=0.7)
#Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, testRatio=0.94)

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)
print("ytrain shape:", ytrain.shape)
print("ytest shape:", ytest.shape)

#Xtrainn, Xvalid, ytrainn, yvalid = splitTrainTestSet(Xtrain, ytrain, 0.1)
Xtrainn, Xvalid, ytrainn, yvalid = splitTrainTestSet(Xtrain, ytrain, 0.1)
print("Xtrainn shape:", Xtrainn.shape)
print("Xvalid shape:", Xvalid.shape)
print("ytrainn shape:", ytrainn.shape)
print("yvalid shape:", yvalid.shape)

import time

start_time = time.time()


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

#Xtrain = Xtrainn.reshape(568,10*25*25)
#Xtest = Xtest.reshape(9627,10*25*25)
#Xvalid = Xvalid.reshape(54,10*25*25)

#Xtrain = Xtrainn.reshape(935,10*25*25)
#Xtest = Xtest.reshape(9218,10*25*25)
#Xvalid = Xvalid.reshape(96,10*25*25)

Xtrain = Xtrainn.reshape(707,10*25*25)
Xtest = Xtest.reshape(9473,10*25*25)
Xvalid = Xvalid.reshape(69,10*25*25)

#Xtrain = Xtrainn.reshape(473,10*25*25)
#Xtest = Xtest.reshape(9729,10*25*25)
#Xvalid = Xvalid.reshape(47,10*25*25)

#Xtrain = Xtrainn.reshape(921,10*25*25)
#Xtest = Xtest.reshape(9225,10*25*25)
#Xvalid = Xvalid.reshape(103,10*25*25)

#Xtrain = Xtrainn.reshape(921,10*15*15)
#Xtest = Xtest.reshape(9225,10*15*15)
#Xvalid = Xvalid.reshape(103,10*15*15)

clf.fit(
    X_train=Xtrain, y_train=ytrainn,
    eval_set=[(Xtrain, ytrainn), (Xvalid, yvalid)],
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

from sklearn.metrics import cohen_kappa_score,accuracy_score
kappa=cohen_kappa_score(ytest,y_pred)
print('kappa coefficient:',kappa)

end_time_test = time.time()
total_time_test = end_time_test - end_time

print(f"Testing completed in {total_time_test:.2f} seconds.")



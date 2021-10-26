import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn import manifold
import torch

def plot_t_SNE(space,labels,domain=None,title=None):
    #Reference: https://blog.csdn.net/hustqb/article/details/80628721
    domain = domain or labels
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=500,n_iter=5000)
    X_tsne = tsne.fit_transform(space)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    plt.title(title)
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1],
                 str(labels[i]),
                 color=plt.cm.Set1(domain[i]), 
                 fontdict={'weight': 'bold', 'size': 9})

def embedding(model,data,predict_func,**kwargs):
    space = predict_func(model,data,**kwargs)
    space = torch.FloatTensor(space).reshape(len(data),-1)
    space = space.tolist()
    return space

def feature_embedding(model,data,labels,title,predict_func,**kwargs):
    space = embedding(model,data,predict_func,**kwargs)
    plot_t_SNE(space,labels,title=title)
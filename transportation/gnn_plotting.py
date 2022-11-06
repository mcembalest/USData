# max_plot_util

## Standard libraries
import os
import json
import math
import numpy as np
import time
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from urllib.request import urlopen
import json
import matplotlib.pyplot as plt

# Plotly
import plotly.graph_objects as go
import plotly.express as px

# NetworkX
import networkx as nx

## PyTorch
import torch

def visualize_graph(ax, G, color=None, labels=False, title=None, layout=lambda G : nx.spring_layout(G, seed=42)):
    ax.set_xticks([])
    ax.set_yticks([])
    node_color = 'white'
    if color is not None: node_color = color
    nx.draw_networkx(G, pos=layout(G), with_labels=labels,
                     node_color=node_color, cmap="Set2", node_size=100, ax=ax)
    if title: ax.set_title(title)

def visualize_embedding(ax, h, color, epoch=None, loss=None, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    h = h.detach().cpu().numpy()
    ax.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        ax.set_xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=10)
    if title: ax.set_title(title)
        
def visualize_model_embedding_spaces(models, modelnames):
    fig, ax = plt.subplots(1, len(models), figsize=(len(models)*4,4))
    x_min, x_max = -1, 1
    y_min, y_max = -1, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    for i, model in enumerate(models):
        if len(models)==1:
            _ax = ax
        else:
            _ax = ax[i]
        assert model.classifier.in_features == 2
        name = modelnames[i]
        pred = model.classifier(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()).detach().numpy()
        zz = np.argmax(pred, axis=1).reshape(xx.shape)
        _ax.pcolormesh(xx, yy, zz)
        _ax.set_title(name)
    plt.tight_layout()
        
def plot_results(losses, acc, modelnames):
    assert list(losses.keys()) == modelnames and list(acc.keys()) == modelnames
    n_models = len(modelnames)
    fig, ax = plt.subplots(2, n_models, figsize=(3*n_models,6))
    if n_models == 1:
        model_acc = acc[modelnames[0]]
        ax[0].set_title(f'{modelnames[0]}: accuracy')
        ax[0].plot(np.arange(len(model_acc)), [x[0] for x in model_acc], label='train acc')
        ax[0].plot(np.arange(len(model_acc)), [x[1] for x in model_acc], label='test acc')
        ax[0].set_xlabel('# epochs')
        ax[0].legend(loc='lower right')
        ax[0].set_ylim(([0,1]))

        model_losses = losses[modelnames[0]]
        ax[1].set_title(f'{modelnames[0]}: loss')
        ax[1].plot(np.arange(len(model_losses)), model_losses)
        ax[1].set_xlabel('# epochs')
        ax[1].set_ylim(([0,max([max(losses[name]) for name in modelnames])]))
    else:
        for i in range(n_models):
            model_acc = acc[modelnames[i]]
            ax[0,i].set_title(f'{modelnames[i]}: accuracy')
            ax[0,i].plot(np.arange(len(model_acc)), [x[0] for x in model_acc], label='train acc')
            ax[0,i].plot(np.arange(len(model_acc)), [x[1] for x in model_acc], label='test acc')
            ax[0,i].set_xlabel('# epochs')
            ax[0,i].legend(loc='lower right')
            ax[0,i].set_ylim(([0,1]))

            model_losses = losses[modelnames[i]]
            ax[1,i].set_title(f'{modelnames[i]}: loss')
            ax[1,i].plot(np.arange(len(model_losses)), model_losses)
            ax[1,i].set_xlabel('# epochs')
            ax[1,i].set_ylim(([0,max([max(losses[name]) for name in modelnames])]))
    plt.tight_layout()

def plot_subgraph_3d(graph, N=100, data=None, names=None):

    subgraph_idx = np.random.choice(np.arange(len(graph.nodes)), N, replace=False)
    subgraph = graph.subgraph(subgraph_idx)
    
    if names is not None: 
        names=names[subgraph_idx]
    if data is not None: 
        subgraph_labels = data.y[subgraph_idx]
    show_labels = data is not None and names is not None  
    hoverlabels = 'none'
    if show_labels: hoverlabels = [f'{n} : {l}' for n, l in zip(names, subgraph_labels)]

    spring_3D = nx.spring_layout(subgraph, dim = 3, k = 0.3) # k regulates the distance between nodes
    nodes3d = [[spring_3D[key][i] for key in spring_3D.keys()] for i in range(3)]
    edges3d = [[], [], []]
    for edge in subgraph.edges():
        for i in range(3):
            edges3d[i] += [spring_3D[edge[0]][i],spring_3D[edge[1]][i],None]
            
    fig = go.Figure(data=[
        go.Scatter3d(x=edges3d[0], y=edges3d[1], z=edges3d[2], 
        mode='lines', 
        line=dict(
            color='black', 
            width=.1), 
        hoverinfo='none'), 
        go.Scatter3d(x=nodes3d[0], y=nodes3d[1], z=nodes3d[2],
        mode='markers',
        marker=dict(
            symbol='circle', 
            size=6, 
            color=subgraph_labels),
        hovertemplate = hoverlabels)
    ])
    fig.show()
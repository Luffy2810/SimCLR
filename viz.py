from src.dataset.dataset import SSLDataset
from src.model.ResnetSimCLR import make_model


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
tsne = TSNE()
import panads as pd


if(os.path.isfile("results/model.pth")):
    resnet.load_state_dict(torch.load("results/model.pth"))
else:
    print("Model Does not exist")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_folder = r'/data'


label_val=pd.read_csv (root_folder+'val_annotations.txt', sep='\t',header=None)[[0,1]]
label_val.columns=['img','label']
label_val=label_val.set_index('img').T.to_dict('list')


training_dataset = MyDataset(root_folder+'/Compiled_train', names_train_10_percent, labels_train_10_percent, train=True,mutate=False)
testing_dataset = MyDataset(root_folder + '/Compiled_val', names_test, label_val, train=False,mutate=False)

dataloader_training_dataset = DataLoader(training_dataset, batch_size=125, shuffle=True, num_workers=2)
dataloader_testing_dataset = DataLoader(testing_dataset, batch_size=250, shuffle=True, num_workers=2)

def plot_vecs_n_labels(v,labels,fname):
    fig = plt.figure(figsize = (10, 10))
    plt.axis('off')
    sns.set_style("darkgrid")
    sns.scatterplot(x=v[:,0], y=v[:,1], hue=labels, legend='full', palette=sns.color_palette("bright", 5))
    plt.savefig(fname)
    plt.close()

TSNEVIS = True

if TSNEVIS:
    resnet.eval()

    for (_, sample_batched) in enumerate(dataloader_training_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_train_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(dataloader_testing_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_test_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-2])

if TSNEVIS:
    for (_, sample_batched) in enumerate(dataloader_training_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_train_second_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(dataloader_testing_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_test_second_last_layer.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

resnet.fc = nn.Sequential(*list(resnet.fc.children())[:-1])

if TSNEVIS:
    for (_, sample_batched) in enumerate(dataloader_training_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_hidden_train.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None

    for (_, sample_batched) in enumerate(dataloader_testing_dataset):
        x = sample_batched['image']
        x = x.to(device)
        y = resnet(x)
        y_tsne = tsne.fit_transform(y.cpu().data)
        labels = sample_batched['label']
        plot_vecs_n_labels(y_tsne,labels,'tsne_hidden_test.png')
        x = None
        y = None
        y_tsne = None
        sample_batched = None 
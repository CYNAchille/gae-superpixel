import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import slic
from scipy.spatial.distance import cdist
import scipy.ndimage
import torch.nn as nn
import torchvision
from skimage import transform
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE,VGAE
from torch_geometric.data import Data
import data_loading
import preprocess
import model
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader


loader =  torch.load('./dataloader.pth')
dataset = loader.dataset



# parameters
out_channels = 2
num_features = dataset[0].x.shape[1]
print('num_features: {:03d}'.format(num_features))
epochs = 300
lr = 0.005

# model
#my_model = GAE(model.GCNEncoder(num_features, out_channels))
my_model = VGAE(model.VariationalGCNEncoder(num_features, out_channels))

# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
my_model = my_model.to(device)


# inizialize the optimizer
optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
def train(model, dataset, variational):
    model.train()
    epoch_loss = 0
    for data in dataset:
        x = data.x.cuda()
        edge = data.edge_index.cuda()
        z = model.encode(x, edge)
        loss = model.recon_loss(z, edge)
        if variational:
           loss +=  (1 / data.num_nodes) * model.kl_loss()
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(epoch_loss / len(dataset))
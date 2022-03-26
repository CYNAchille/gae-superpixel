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
from torch_geometric.nn import GAE, VGAE
import model
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from plot_loss import plot_losses

train_loader = torch.load('./train_dataloader.pth')
train_dataset = train_loader.dataset
valid_loader = torch.load('./valid_dataloader.pth')
valid_dataset = valid_loader.dataset

# parameters
out_channels = 2
num_features = train_dataset[0].x.shape[1]
print('num_features: {:03d}'.format(num_features))
epochs = 10
lr = 0.001

# model
# my_model = GAE(model.GCNEncoder(num_features, out_channels))
my_model = VGAE(model.VariationalGCNEncoder(num_features, out_channels))
my_model.reset_parameters()

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
            loss += (1 / data.num_nodes) * model.kl_loss()
        epoch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(epoch_loss / len(dataset))


def test(model, dataset, variational):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for data in dataset:
            x = data.x.cuda()
            edge = data.edge_index.cuda()
            z = model.encode(x, edge)
            loss = model.recon_loss(z, edge)
            if variational:
                loss += (1 / data.num_nodes) * model.kl_loss()
            epoch_loss += loss
    return float(epoch_loss / len(dataset))

train_history = []
valid_history = []
for epoch in range(1, epochs + 1):
    loss_train = train(my_model, train_dataset, True)
    loss_valid = test(my_model, valid_dataset, True)
    train_history.append(loss_train)
    valid_history.append(loss_valid)
    print('Epoch: {:03d}, mean loss: {:.4f}, valid loss: {:.4f}'.format(epoch, loss_train, loss_valid))

data = train_dataset[4]
x = data.x.cuda()
edge = data.edge_index.cuda()
z = my_model.encode(x, edge).cpu().detach().numpy()
print(z.shape)
'''
tsne = TSNE(n_components=2).fit_transform(z)
plt.scatter(tsne[:,0],tsne[:,1])
plt.show()
'''
plt.subplot(121)
plt.scatter(z[:, 0], z[:, 1])
plt.subplot(122)
plot_losses(train_history, valid_history)
torch.save(my_model.state_dict(), './checkpoint_model1.pth')

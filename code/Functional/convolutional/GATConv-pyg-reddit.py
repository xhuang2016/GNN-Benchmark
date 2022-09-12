import os
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
# from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
from torch_sparse import SparseTensor
import time
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
import torch_geometric

# log_name = 'pyg-reddit'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/reddit/"))

dataset = Reddit(path)
data = dataset[0]

data.x = data.x.to(device)
data.edge_index, _ = add_self_loops(data.edge_index)
data.edge_index = data.edge_index.to(device)
adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])

conv = torch_geometric.nn.GATConv(dataset.num_features, 256, heads=1, concat=True, negative_slope = 0.2, dropout = 0.0, 
                                  add_self_loops = False, edge_dim = None, fill_value = 'mean', bias = True).to(device)

if device == torch.device('cuda'):
    #warm up
    for i in range(20):
        res = conv(data.x, adj)
        torch.cuda.synchronize()

    for i in range(10):
        t = time.perf_counter()
        res = conv(data.x, adj)
        torch.cuda.synchronize()
        print(time.perf_counter()-t)

if device == torch.device('cpu'): 
    #warm up
    for i in range(5):
        res = conv(data.x, adj)
    for i in range(10):
        t = time.perf_counter()
        res = conv(data.x, adj)
        print(time.perf_counter()-t)


print('====================================')
    
# tracker.stop()
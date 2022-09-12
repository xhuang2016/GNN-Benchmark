import os
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, add_self_loops
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time
from torch_sparse import SparseTensor
import torch_geometric

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=path)
data = dataset[0]
n_classes = dataset.num_classes
data.edge_index = to_undirected(data.edge_index, data.num_nodes)

# data.x = data.x.to(device)
# data.edge_index = data.edge_index.to(device)
# adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
x = data.x
y = data.y.squeeze()

split_idx = dataset.get_idx_split()

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask


log_name = 'pyg-arxiv'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
energy_logger(log_name)
tracker.start()

for i in range(11):
    t = time.perf_counter()
    train_loader = torch_geometric.loader.NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                sizes=[25, 10], batch_size=512, shuffle=True, num_workers=0)
    print(time.perf_counter()-t)
    for batch_size, n_id, adjs in train_loader:
        # adjs = [adj.to(device) for adj in adjs]
        feat = x[n_id]
        label = y[n_id[:batch_size]]

    if i>0:
        print(time.perf_counter()-t)

print('====================================')
    
tracker.stop()
import os
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_undirected, add_self_loops
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time
from torch_sparse import SparseTensor
import torch_geometric

# log_name = 'pyg-ogbn-products'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))
for i in range(10):
    t = time.perf_counter()
    dataset = PygNodePropPredDataset(name='ogbn-products', root=path)
    data = dataset[0]
    n_classes = dataset.num_classes
    print(time.perf_counter()-t)

# data.x = data.x.to(device)
# data.edge_index = data.edge_index.to(device)
# adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])

print('====================================')
    
# tracker.stop()

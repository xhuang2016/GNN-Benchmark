import os
import torch
from torch_geometric.datasets import Yelp
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
import torch_geometric

# log_name = 'pyg-yelp'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/yelp/"))

dataset = Yelp(path)
data = dataset[0]

data.x = data.x.to(device)
data.edge_index, _ = add_self_loops(data.edge_index)
data.edge_index = data.edge_index.to(device)
adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])

conv = torch_geometric.nn.SGConv(dataset.num_features, 256, K = 1, cached = False, 
                                 add_self_loops = False, bias = True).to(device)

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
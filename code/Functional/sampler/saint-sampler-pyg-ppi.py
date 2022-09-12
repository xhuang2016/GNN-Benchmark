import os
import torch
from create_pyg import PPI
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
import torch_geometric


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/ppi/"))

dataset = PPI(path)
data = dataset[0]
n_classes = dataset.num_classes

# data.edge_index = data.edge_index.to(device)
# adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
x = data.x
y = data.y.squeeze()

log_name = 'pyg-ppi'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
energy_logger(log_name)
tracker.start()

iter = (torch.count_nonzero(data.train_mask)/(3000*2)).int()
# print(iter)

for i in range(15):
    t = time.perf_counter()

    loader = torch_geometric.loader.GraphSAINTRandomWalkSampler(data, batch_size=3000, walk_length=2, num_steps=iter, 
                                                                sample_coverage=0,save_dir=None,num_workers=0)
    # print(time.perf_counter()-t)
    for batch in loader:
        x=batch.x
        y=batch.y
        adj=batch.edge_index

    if i> 4:
        print(time.perf_counter()-t)

print('====================================')


tracker.stop()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time


# log_name = 'dgl-ogbn-products'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))
for i in range(10):
    t = time.perf_counter()
    dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset('ogbn-products', root=path))
    graph = dataset[0]
    n_classes = dataset.num_classes
    print(time.perf_counter()-t)

# graph = graph.to(device)
# features = graph.ndata['feat'].to(device)

print('====================================')
    
# tracker.stop()

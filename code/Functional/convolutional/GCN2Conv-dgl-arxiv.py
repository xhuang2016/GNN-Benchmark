import os
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
# from experiment_impact_tracker.compute_tracker import ImpactTracker
import time


# log_name = 'dgl-arxiv'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset('ogbn-arxiv', root=path))
graph = dataset[0]
graph = dgl.to_bidirected(graph,copy_ndata=True)
graph = dgl.add_self_loop(graph)

graph = graph.to(device)
features = graph.ndata['feat'].to(device)

conv = dgl.nn.GCN2Conv(graph.ndata['feat'].shape[1], layer=1, alpha=0.1, lambda_=1, project_initial_features=True, 
                       allow_zero_in_degree=False, bias=True, activation=None).to(device)

if device == torch.device('cuda'):
    #warm up
    for i in range(20):
        res = conv(graph, features, features)
        torch.cuda.synchronize()

    for i in range(10):
        t = time.perf_counter()
        res = conv(graph, features, features)
        torch.cuda.synchronize()
        print(time.perf_counter()-t)

if device == torch.device('cpu'):
    #warm up
    for i in range(5):
        res = conv(graph, features, features)
    for i in range(10):
        t = time.perf_counter()
        res = conv(graph, features, features)
        print(time.perf_counter()-t)

print('====================================')
    
# tracker.stop()

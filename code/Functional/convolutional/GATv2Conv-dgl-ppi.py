import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score
from load_graph import load_ppi
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time

# log_name = 'dgl-ppi'
# tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
# energy_logger(log_name)
# tracker.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
   
path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

graph, num_classes = load_ppi()
graph = dgl.add_self_loop(graph)

graph = graph.to(device)
features = graph.ndata['feat'].to(device)

conv = dgl.nn.GATv2Conv(graph.ndata['feat'].shape[1], 256, 1, feat_drop=0.0, attn_drop=0.0, negative_slope=0.2, 
                      residual=False, activation=None, allow_zero_in_degree=False, bias=True, share_weights=False).to(device)

if device == torch.device('cuda'):
    #warm up
    for i in range(20):
        res = conv(graph, features)
        torch.cuda.synchronize()

    for i in range(10):
        t = time.perf_counter()
        res = conv(graph, features)
        torch.cuda.synchronize()
        print(time.perf_counter()-t)

if device == torch.device('cpu'):
    #warm up
    for i in range(5):
        res = conv(graph, features)
    for i in range(10):
        t = time.perf_counter()
        res = conv(graph, features)
        print(time.perf_counter()-t)

print('====================================')
    
# tracker.stop()
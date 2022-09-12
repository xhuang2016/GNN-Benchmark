import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score
from load_graph import load_yelp
from codecarbon import EmissionsTracker
from energy_logger import energy_logger
import time

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

graph, num_classes = load_yelp()

# graph = graph.to(device)
# features = graph.ndata['feat'].to(device)
in_feats = graph.ndata['feat'].shape[1]

train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0].to(device)
val_idx = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0].to(device)
test_idx = torch.nonzero(~(graph.ndata['train_mask'] | graph.ndata['val_mask']), as_tuple=True)[0].to(device)

log_name = 'dgl-yelp'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='emissions.csv',)
energy_logger(log_name)
tracker.start()

iter = (torch.count_nonzero(graph.ndata['train_mask'])/(3000*2)).int()

for i in range(105):
    t = time.perf_counter()
    sampler = dgl.dataloading.SAINTSampler(mode='walk', budget=[3000, 2], cache=False)
    dataloader = dgl.dataloading.DataLoader(graph, torch.arange(iter), sampler, batch_size=1, num_workers=0, 
                                            device='cpu', shuffle=True, drop_last=False, use_uva=False)
    for sg in dataloader:
        x = sg.ndata['feat']
        y = sg.ndata['label']

    if i > 4:
        print(time.perf_counter()-t)

print('====================================')
    
tracker.stop()

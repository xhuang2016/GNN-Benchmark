import os
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from energy_logger import energy_logger


class GraphSAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))
        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.0, training=self.training)
        return F.log_softmax(x, dim=-1)
    
    def inference(self, x_all):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all


def train():
    model.train()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss = F.cross_entropy(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.valid_mask, data.test_mask]:
        f1_micro = f1_score(data.y[mask].cpu(), y_pred[mask].cpu(), average='micro')
        accs.append(f1_micro)  

    print('Training acc:', accs[0],  'Validation acc:', accs[1], 'Testing acc:', accs[2])


log_name = 'pyg_graphsaint-ogbn-arxiv-cpu'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAINT-emissions.csv')
energy_logger(log_name)
tracker.start()

device = torch.device('cpu')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

dataset = PygNodePropPredDataset('ogbn-arxiv', root = path)
split_idx = dataset.get_idx_split()
# evaluator = Evaluator(name='ogbn-arxiv')

data = dataset[0]
train_idx = split_idx['train']

for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask

data.edge_index = to_undirected(data.edge_index, data.num_nodes)
# row, col = data.edge_index
# data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

iter = (torch.count_nonzero(data.train_mask)/(3000*3)).int()
# print(iter)

loader = GraphSAINTRandomWalkSampler(data, batch_size=3000, walk_length=2, num_steps=iter, sample_coverage=0,save_dir=None,num_workers=0)

subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=0)

model = GraphSAINT(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


print('PyG, ogbn-arxiv, GraphSAINT, cpu')
for epoch in range(100):
    train()
print('Training done!')
# test()
# print('===============================')
tracker.stop()
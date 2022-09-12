import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from codecarbon import EmissionsTracker
from energy_logger import energy_logger


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_layers,
                 dropout):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats, aggr='mean'))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggr='mean'))
        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, adj)

        return x.log_softmax(dim=-1)


def train(model, x, adj, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj)[train_idx]
    # loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss = F.cross_entropy(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    # return loss.item()

@torch.no_grad()
def test():  # Inference should be performed on the full graph.
    model.eval()

    out = model(x, adj)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.valid_mask, data.test_mask]:
        f1_micro = f1_score(data.y[mask].cpu(), y_pred[mask].cpu(), average='micro')
        accs.append(f1_micro)

    print('Training acc:', accs[0],  'Validation acc:', accs[1], 'Testing acc:', accs[2])



parser = argparse.ArgumentParser(description='OGBN-Products (GraphSAGE Full-Batch)')
parser.add_argument("--dataset", type=str, default='products')
parser.add_argument('--device', type=int, default=0)
# parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument("--eval", action='store_true',
                    help='If not set, we will only do the training part.')
args = parser.parse_args()
print(args)

# device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# device = torch.device(device)

log_name = 'pyg_graphsage-full-batch-ogbn-products-' + device
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAGE-fullbatch-emissions.csv',)
energy_logger(log_name)
tracker.start()

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

dataset = PygNodePropPredDataset(name='ogbn-products', root=path)
split_idx = dataset.get_idx_split()

data = dataset[0]
edge_index = data.edge_index.to(device)
# edge_index = to_undirected(edge_index, data.num_nodes)
adj = SparseTensor(row=edge_index[0], col=edge_index[1])


for key, idx in split_idx.items():
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[idx] = True
    data[f'{key}_mask'] = mask


x, y_true = data.x.to(device), data.y.to(device)
train_idx = split_idx['train'].to(device)

model = GraphSAGE(in_feats=data.x.size(-1),
                    hidden_feats=args.hidden_channels,
                    out_feats=dataset.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout).to(device)

# evaluator = Evaluator(name='ogbn-products')

print('PyG, products, GraphSAGE, full-batch, ' + device)
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, 1 + args.epochs):
        train(model, x, adj, y_true, train_idx, optimizer)
    print('Training done!')
    # test()
    # print('===============================')
tracker.stop()
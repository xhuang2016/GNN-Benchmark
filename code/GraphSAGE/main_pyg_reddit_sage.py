import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
from torch_geometric.datasets import Reddit, Reddit2, Flickr, Yelp
from codecarbon import EmissionsTracker
from energy_logger import energy_logger



class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggr,
                 dropout=0.):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggr))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggr))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        h = x
        for _, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.layers[-1](h, edge_index)
        # return h
        return F.log_softmax(h, dim=-1)


def train(model, x, adj, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj)[train_idx]
    # loss_fcn = nn.MultiLabelSoftMarginLoss()
    # loss_fcn = nn.BCEWithLogitsLoss()
    # loss = F.nll_loss(out, y_true[train_idx])
    loss = F.cross_entropy(out, y_true[train_idx])
    # loss = loss_fcn(out, y_true[train_idx].float())
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()

    out = model(features, adj)
    y_pred = out.argmax(dim=-1)

    y_label = data.y
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        f1_micro = f1_score(y_label[mask].cpu(), y_pred[mask].cpu(), average='micro')
        accs.append(f1_micro)

    print('Training acc:', accs[0],  'Validation acc:', accs[1], 'Testing acc:', accs[2])



parser = argparse.ArgumentParser(description='Reddit (GraphSAGE Full-Batch)')
parser.add_argument("--dataset", type=str, default='reddit')
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0,
                    help="dropout probability")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="learning rate")
parser.add_argument("--epochs", type=int, default=100,
                    help="number of training epochs")
parser.add_argument("--n-hidden", type=int, default=256,
                    help="number of hidden gcn units")
parser.add_argument("--aggr", type=str, choices=['sum', 'mean'], default='mean',
                    help='Aggregation for messages')
parser.add_argument("--weight-decay", type=float, default=0,
                    help="Weight for L2 loss")
parser.add_argument("--eval", action='store_true',
                    help='If not set, we will only do the training part.')
parser.add_argument("--runs", type=int, default=1)
args = parser.parse_args()
print(args)


# device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
# device = torch.device(device)

log_name = 'pyg_graphsage-full-batch-reddit-' + device
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAGE-fullbatch-emissions.csv',)
energy_logger(log_name)
tracker.start()

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/reddit/"))
dataset = Reddit(path)
data = dataset[0]

features = data.x.to(device)
labels = data.y.to(device)
edge_index = data.edge_index.to(device)
adj = SparseTensor(row=edge_index[0], col=edge_index[1])
train_mask = torch.BoolTensor(data.train_mask).to(device)
val_mask = torch.BoolTensor(data.val_mask).to(device)
test_mask = torch.BoolTensor(data.test_mask).to(device)

model = GraphSAGE(dataset.num_features,
                    args.n_hidden,
                    dataset.num_classes,
                    args.aggr,
                    args.dropout).to(device)

print('PyG, reddit, GraphSAGE, full-batch, ' + device)
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(1, 1 + args.epochs):
        train(model, features, adj, labels, train_mask, optimizer)
    print('Training done!')
    # test()
    # print('===============================')
tracker.stop()
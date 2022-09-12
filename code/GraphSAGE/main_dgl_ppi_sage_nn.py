import os
import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import f1_score
from load_graph import load_ppi
from codecarbon import EmissionsTracker
from energy_logger import energy_logger


class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggr,
                 activation=F.relu,
                 dropout=0.):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.g = g
        self.layers.append(SAGEConv(in_feats, n_hidden, aggr, activation=activation))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggr, feat_drop=dropout, activation=None))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(self.g, h)
        # return h
        return F.log_softmax(h, dim=-1)


def train(model, feats, y_true, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(feats)[train_idx]
    # loss_fcn = nn.MultiLabelSoftMarginLoss()
    loss_fcn = nn.BCEWithLogitsLoss()
    loss = loss_fcn(out, y_true[train_idx].float())
    loss.backward()
    optimizer.step()

    # return loss.item()


@torch.no_grad()
def test():
    # model.to('cpu')
    model.eval()

    m_train = train_mask
    m_val = val_mask
    m_test = test_mask
    y_hat = model(features)

    train_preds = y_hat[m_train]
    train_labels = labels[m_train]
    val_preds = y_hat[m_val]
    val_labels = labels[m_val]
    test_preds = y_hat[m_test]
    test_labels = labels[m_test]

    train_preds[train_preds > 0] = 1
    train_preds[train_preds <= 0] = 0
    val_preds[val_preds > 0] = 1
    val_preds[val_preds <= 0] = 0
    test_preds[test_preds > 0] = 1
    test_preds[test_preds <= 0] = 0
    train_acc = f1_score(train_preds.cpu(), train_labels.cpu(), average='micro')
    val_acc = f1_score(val_preds.cpu(), val_labels.cpu(), average='micro')
    test_acc = f1_score(test_preds.cpu(), test_labels.cpu(), average='micro')
    print('Training acc:', train_acc.item(),  'Validation acc:', val_acc.item(), 'Testing acc:', test_acc.item())



parser = argparse.ArgumentParser(description='PPI (GraphSAGE Full-Batch)')
parser.add_argument("--dataset", type=str, default='ppi')
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

log_name = 'dgl_graphsage-full-batch-ppi-' + device
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAGE-fullbatch-emissions.csv',)
energy_logger(log_name)
tracker.start()

# load and preprocess dataset
path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

graph, n_classes = load_ppi()

features =  torch.FloatTensor(graph.ndata['feat'])
labels = torch.LongTensor(graph.ndata['labels'])

in_feats = features.shape[1]

if hasattr(torch, 'BoolTensor'):
    train_mask = torch.BoolTensor(graph.ndata['train_mask'])
    val_mask = torch.BoolTensor(graph.ndata['val_mask'])
    test_mask = torch.BoolTensor(graph.ndata['test_mask'])
else:
    train_mask = torch.ByteTensor(graph.ndata['train_mask'])
    val_mask = torch.ByteTensor(graph.ndata['val_mask'])
    test_mask = torch.ByteTensor(graph.ndata['test_mask'])

features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

# g = graph.int().to(device)
g = graph.to(device)

# create GraphSAGE model
model = GraphSAGE(g, in_feats, args.n_hidden, n_classes, args.aggr, F.relu, args.dropout).to(device)

print('DGL, ppi, GraphSAGE, full-batch, ' + device)
for run in range(args.runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, 1 + args.epochs):
        train(model, features, labels, train_mask, optimizer)
    print('Training done!')
    # test()
    # print('===============================')
tracker.stop()
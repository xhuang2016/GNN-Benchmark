import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit, Flickr, Yelp
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from sklearn.metrics import f1_score
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from create_pyg import PPI
from codecarbon import EmissionsTracker
from energy_logger import energy_logger

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        self.num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))
        self.dropout = torch.nn.Dropout(p=0.0)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)


        return x_all


def train(epoch):
    model.train()

    for batch_size, n_id, adjs in train_loader:

        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.cross_entropy(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()



@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    accs = []

    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        f1_micro = f1_score(y_true[mask], y_pred[mask], average='micro')
        accs.append(f1_micro)
    print('Training acc:', accs[0],  'Validation acc:', accs[1], 'Testing acc:', accs[2])


log_name = 'pyg_graphsage-reddit-cpugpu'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAGE-emissions.csv',)
energy_logger(log_name)
tracker.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/reddit/"))
dataset = Reddit(path)
data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[25, 10], batch_size=512, shuffle=True,
                               num_workers=0)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=512, shuffle=False,
                                  num_workers=0)

model = GraphSAGE(dataset.num_features, 256, dataset.num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x = data.x.to(device)
y = data.y.squeeze().to(device)


print('PyG, reddit, GraphSAGE, mini-batch, CPU-sampling & GPU-training')
for epoch in range(10):
    train(epoch)
print('Training done!')
test()
print('===============================')
tracker.stop()
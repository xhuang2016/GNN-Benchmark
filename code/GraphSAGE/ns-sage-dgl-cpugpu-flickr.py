import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from load_graph import load_reddit, load_ogb, inductive_split, load_ppi, load_flickr, load_yelp
from sklearn.metrics import f1_score
from codecarbon import EmissionsTracker
from energy_logger import energy_logger


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dgl.nn.SAGEConv(in_feats, n_hidden, 'mean'))
        # self.layers.append(dgl.nn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dgl.nn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(0.0)
        self.n_hidden = n_hidden
        self.n_classes = n_classes

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        # return h
        return F.log_softmax(h, dim=-1)

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        feat = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=num_workers)

        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device, pin_memory=True)
            feat = feat.to(device)
            for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def train():
    model.train()
    for _, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        x = blocks[0].srcdata['feat'].to('cuda')
        y = blocks[-1].dstdata['label'].to('cuda')
        blocks = [block.int().to('cuda') for block in blocks]
        y_hat = model(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        # loss = nn.BCEWithLogitsLoss()
        # loss = loss(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def test():
    model.to('cpu')
    model.eval()

    m_train = graph.ndata['train_mask'].bool()
    m_val = graph.ndata['val_mask'].bool()
    m_test = graph.ndata['test_mask'].bool()
    y_hat = model.inference(graph, 'cpu', 512, 0, 'cpu')

    train_preds = y_hat[m_train]
    train_labels = graph.ndata['label'][m_train]

    val_preds = y_hat[m_val]
    val_labels = graph.ndata['label'][m_val]

    test_preds = y_hat[m_test]
    test_labels = graph.ndata['label'][m_test]

    train_preds = train_preds.argmax(dim=-1)
    val_preds = val_preds.argmax(dim=-1)
    test_preds = test_preds.argmax(dim=-1)
    train_acc = f1_score(train_preds.cpu(), train_labels.cpu(), average='micro')
    val_acc = f1_score(val_preds.cpu(), val_labels.cpu(), average='micro')
    test_acc = f1_score(test_preds.cpu(), test_labels.cpu(), average='micro')
    print('Training acc:', train_acc.item(),  'Validation acc:', val_acc.item(), 'Testing acc:', test_acc.item())


log_name = 'dgl_graphsage-flickr-cpugpu'
tracker = EmissionsTracker(measure_power_secs=0.1, project_name=log_name, output_dir='log/', output_file='GraphSAGE-emissions-dgl.csv',)
energy_logger(log_name)
tracker.start()

path = os.path.abspath(os.path.join(os.getcwd(), "../dataset/"))

graph, num_classes = load_flickr()

in_feats = graph.ndata['feat'].shape[1]

train_idx = torch.nonzero(graph.ndata['train_mask'], as_tuple=True)[0].to('cpu')
val_idx = torch.nonzero(graph.ndata['val_mask'], as_tuple=True)[0].to('cpu')
test_idx = torch.nonzero(~(graph.ndata['train_mask'] | graph.ndata['val_mask']), as_tuple=True)[0].to('cpu')

graph = graph.to('cpu')

model = GraphSAGE(graph.ndata['feat'].shape[1], 256, num_classes).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


sampler = dgl.dataloading.NeighborSampler([25, 10])
train_dataloader = dgl.dataloading.DataLoader(
        graph, train_idx, sampler, device='cpu', batch_size=512, shuffle=True,
        drop_last=False, num_workers=0, use_uva=False)
# valid_dataloader = dgl.dataloading.DataLoader(
#         graph, torch.arange(graph.num_nodes()).to('cpu'), sampler, device='cpu', batch_size=512, shuffle=True,
#         drop_last=False, num_workers=0, use_uva=False)


print('DGL, flickr, GraphSAGE, mini-batch, cpu-gpu without Prefetching')
for epoch in range(1, 11):
    train()
print('Training done!')
# test()
# print('===============================')
tracker.stop()
import dgl
from dgl.data import DGLDataset
import torch
import json
import numpy as np
import scipy.sparse
import os

class PPIDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='ppi')

    def process(self):
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        adj_full = scipy.sparse.load_npz(path+'/data/ppi/adj_full.npz').astype('bool')
        self.graph = dgl.from_scipy(adj_full)
        feats = np.load(path+"/data/ppi/feats.npy")
        labels = json.load(open(path+"/data/ppi/class_map.json"))

        role = json.load(open(path+"/data/ppi/role.json"))

        node_features = torch.from_numpy(feats).float()
        node_labels = torch.from_numpy(np.array(list(labels.values()), dtype=np.int64))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)

        train_ids = role['tr']
        test_ids = role['te']
        val_ids = role['va']

        for i in train_ids:
            train_mask[i] = True

        for i in test_ids:
            test_mask[i] = True

        for i in val_ids:
            val_mask[i] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

# dataset = PPIDataset()
# graph = dataset[0]

# print(graph)


class flickrDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='flickr')

    def process(self):
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        adj_full = scipy.sparse.load_npz(path+'/data/flickr/adj_full.npz').astype('bool')
        self.graph = dgl.from_scipy(adj_full)
        feats = np.load(path+"/data/flickr/feats.npy")
        labels = json.load(open(path+"/data/flickr/class_map.json"))

        role = json.load(open(path+"/data/flickr/role.json"))

        node_features = torch.from_numpy(feats).float()
        node_labels = torch.from_numpy(np.array(list(labels.values()), dtype=np.int64))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)

        train_ids = role['tr']
        test_ids = role['te']
        val_ids = role['va']

        for i in train_ids:
            train_mask[i] = True

        for i in test_ids:
            test_mask[i] = True

        for i in val_ids:
            val_mask[i] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1





class yelpDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='yelp')

    def process(self):
        path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        adj_full = scipy.sparse.load_npz(path+'/data/yelp/adj_full.npz').astype('bool')
        self.graph = dgl.from_scipy(adj_full)
        feats = np.load(path+"/data/yelp/feats.npy")
        labels = json.load(open(path+"/data/yelp/class_map.json"))

        role = json.load(open(path+"/data/yelp/role.json"))

        node_features = torch.from_numpy(feats).float()
        node_labels = torch.from_numpy(np.array(list(labels.values()), dtype=np.int64))
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        train_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(self.graph.num_nodes(), dtype=torch.bool)

        train_ids = role['tr']
        test_ids = role['te']
        val_ids = role['va']

        for i in train_ids:
            train_mask[i] = True

        for i in test_ids:
            test_mask[i] = True

        for i in val_ids:
            val_mask[i] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
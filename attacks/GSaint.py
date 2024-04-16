"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
Credit: https://mlabonne.github.io/blog/posts/2022-04-06-GraphSAGE.html
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from torch_geometric.nn import GraphSAGE, SAGEConv, GraphConv
from torch_geometric.loader import NeighborLoader
from deeprobust.graph.data import Dpr2Pyg, Pyg2Dpr
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.typing import WITH_TORCH_SPARSE
from torch_geometric.utils import degree


def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()


class GSAINT(nn.Module):
    """
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    nlayers : int
        number of layers
    dropout : float
        dropout rate for GAT
    lr : float
        learning rate for GAT
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in GAT weights.
    device: str
        'cpu' or 'cuda'.
    """

    #     model = GraphSAGE(
    #     data.num_node_features,
    #     hidden_channels=64,
    #     num_layers=2,
    # ).to(device)

    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        dropout=0.5,
        lr=0.01,
        weight_decay=5e-4,
        with_relu=False,
        with_bias=True,
        device=None,
    ):

        super(GSAINT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = GraphConv(nfeat, nhid)
        self.conv2 = GraphConv(nhid, nhid)
        self.conv3 = GraphConv(nhid, nhid)
        self.lin = torch.nn.Linear(3 * nhid, nclass)

        # self.sage1 = SAGEConv(nfeat, nhid)
        # self.sage2 = SAGEConv(nhid, nclass)

        self.nclass = nclass
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.with_relu = with_relu

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None

        self.optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, data):
        data = data.to(self.device)
        x0, edge_index = data.x, data.edge_index

        # row, col = self.data.edge_index
        edge_weight = data.edge_weight
        # edge_weight = 1.0 / degree(col, data.num_nodes)[col]

        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)
        # x, edge_index = data.x, data.edge_index

        # h = self.sage1(x, edge_index).relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        # h = self.sage2(h, edge_index)
        # return F.log_softmax(h, dim=1)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.conv2(x, edge_index)
        # return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GSAINT."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin.reset_parameters()

    def fit(
        self,
        data,
        train_iters=1000,
        initialize=True,
        verbose=False,
        patience=100,
        **kwargs
    ):
        """Train the GSAINT model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        pyg_data = Dpr2Pyg(data)

        if initialize:
            self.initialize()

        self.data = pyg_data[0]

        row, col = self.data.edge_index
        self.data.edge_weight = 1. / degree(col, self.data.num_nodes)[col]

        self.train_loader = GraphSAINTRandomWalkSampler(
            self.data,
            batch_size=6000,
            walk_length=2,
            num_steps=5,
            sample_coverage=100,
        )

        # self.train_loader = NeighborLoader(
        #     self.data,
        #     num_neighbors=[5, 10],
        #     batch_size=16,
        #     input_nodes=self.data.train_mask,
        # )

        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss"""
        if verbose:
            print("=== training GSAINT model ===")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer

        self.train()
        self.set_aggr("add")

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        # total_loss = 0
        # acc = 0
        # val_loss = 0
        # val_acc = 0
        for i in range(train_iters):
          total_loss = total_examples = 0
          for batch in self.train_loader:
              # self.train()
              batch = batch.to(self.device)
              optimizer.zero_grad()

              edge_weight = batch.edge_norm * batch.edge_weight
              out = self.forward(batch)
              loss = F.nll_loss(out, batch.y, reduction="none")
              loss = (loss * batch.node_norm)[batch.train_mask].sum()

              loss.backward()
              optimizer.step()
              total_loss += loss.item() * batch.num_nodes
              total_examples += batch.num_nodes
              
          if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss.item()))

          self.eval()
          output = self.forward(self.data)
          loss_val = criterion(output[self.data.val_mask], self.data.y[self.data.val_mask])

          if best_loss_val > loss_val:
            best_loss_val = loss_val
            self.output = output
            weights = deepcopy(self.state_dict())
            patience = early_stopping
          else:
            patience -= 1
            if i > early_stopping and patience <= 0:
              break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self):
        """Evaluate GSAINT performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()

        self.set_aggr("mean")

        out = self.forward(
            self.data
        )
        pred = out.argmax(dim=-1)
        correct = pred.eq(self.data.y.to(self.device))

        accs = []
        for _, mask in self.data("test_mask"):
            accs.append(correct[mask].sum().item() / mask.sum().item())
        return accs[0]

        # criterion = torch.nn.CrossEntropyLoss()
        # test_mask = self.data.test_mask
        # labels = self.data.y
        # output = self.forward(self.data)

        # # loss_val = criterion(output[self.data.val_mask], self.data.y[self.data.val_mask])
        # # output = self.output
        # loss_test = criterion(output[test_mask], labels[test_mask])
        # acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        # return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GAT
        """

        self.eval()
        return self.forward(self.data)


# if __name__ == "__main__":
# from deeprobust.graph.data import Dataset, Dpr2Pyg
# # from deeprobust.graph.defense import GAT
# data = Dataset(root='/tmp/', name='cora')
# adj, features, labels = data.adj, data.features, data.labels
# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# gat = GSAGE(nfeat=features.shape[1],
#       nhid=8, heads=8,
#       nclass=labels.max().item() + 1,
#       dropout=0.5, device='cpu')
# gat = gat.to('cpu')
# pyg_data = Dpr2Pyg(data)
# gat.fit(pyg_data, verbose=True) # train with earlystopping
# gat.test()
# print(gat.predict())

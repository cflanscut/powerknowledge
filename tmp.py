from graph.PLAID_dataset import PLAIDDataset
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
import os.path as osp
path = osp.join(osp.abspath('.'), 'graph', 'PLAIDG')
datasets = PLAIDDataset(root=path,
                        name='PLAIDG',
                        use_node_attr=True,
                        use_edge_attr=True)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(
            datasets.num_node_features + datasets.num_edge_features, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        node_slice = self.node_4edge_slice(edge_index)
        for i in range(len(x)):
            tmp = edge_attr[node_slice[i]:node_slice[i + 1]]
            tmp = torch.sum(tmp, 0)
            tmp = torch.unsqueeze(tmp, -1)
            if i == 0:
                x_edge = tmp
                continue
            x_edge = torch.cat([x_edge, tmp], 0)
        x = torch.cat([x, x_edge], 1)
        x.to(device)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x

    def node_4edge_slice(self, edge_index):
        row, _ = edge_index[0], edge_index[1]
        row = row.cpu().numpy()
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(row)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])
        return node_slice


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = datasets.data.to(device)
loader = DataLoader(datasets, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
optimizer.zero_grad()
for batch in loader:
    batch.to(device)
    # print(batch)
    # print(batch.batch)
    # print(batch.num_graphs)
    # print(batch.x.shape)
    # sum = 0
    # for i, value in enumerate(batch.edge_index[0]):
    #     sum += batch.edge_attr[i].numpy()
    #     print('{}:{}，sum={}'.format(value, batch.edge_attr[i], sum))
    out = model(batch)
    # print(batch.edge_index[:, 50])
    # print(batch.edge_attr[50])
    break

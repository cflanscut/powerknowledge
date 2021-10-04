import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.data import DataLoader

# 数据集加载
# from torch_geometric.datasets import Planetoid
# dataset = Planetoid(root='/tmp/Cora', name='Cora')
from PLAID_dataset import PLAIDDataset
datasets = PLAIDDataset(root='~/powerknowledge/graph/PLAIDG',
                        name='PLAIDG',
                        use_node_attr=True,
                        use_edge_attr=True)

# 网络定义
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(datasets.num_node_features, 16) # 节点特征为num_node_feature,经过计算后变成16
#         self.conv2 = GCNConv(16, datasets.num_classes)   # 节点特征为16，经过计算变成类别数（再softmax）

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)  # 计算不影响图的结构，只影响节点的值
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)


class DGCNN(torch.nn.Module):
    def __init__(
            self,
            output_dim,
            num_node_feats,
            num_edge_feats,
            latent_dim=[32, 32, 32, 1],
            k=30,
            conv1d_channels=[16, 32],
            conv1d_kws=[0, 5],  # 0是待定的意思
            conv1d_activation='ReLU'):
        super(DGCNN, self).__init__()

        self.latent_dim = latent_dim  # 指的是图卷积里面的每个层的节点特征维度，最后一层为1，论文写着方便排序
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats  # dataset.num_node_features
        self.num_edge_feats = num_edge_feats  # dataset.num_edge_features
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim  # 把所有图层拼接起来作为1-D卷积的输入

        # GCN
        self.conv1 = GCNConv(self.num_node_feats + self.num_edge_feats,
                             self.latent_dim[0])
        self.conv2 = GCNConv(self.latent_dim[0], self.latent_dim[1])
        self.conv3 = GCNConv(self.latent_dim[1], self.latent_dim[2])
        self.conv4 = GCNConv(self.latent_dim[2], self.latent_dim[3])

        # conv1d
        self.conv1d_params1 = nn.Conv1d(
            1, conv1d_channels[0], conv1d_kws[0],
            conv1d_kws[0])  # 一维输入则通道为1，所有节点特征拼成1维了，核长等于步长，一步一个节点
        self.maxpool1d = nn.MaxPool1d(2, 2)  # 窗口大小2，步长2，沿着一维的方向(类似于两个节点的输出取最大)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
                                        conv1d_kws[1], 1)

        dense_dim = int(
            (k - 2) / 2 +
            1)  # 这个跟池化的输出有关。如100个节点卷完第一层后变成100个数，再池化成50个节点（每个节点有embedding维度）
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[
            1]  # 一维的长度是池化后再一维卷，即dense-kws+1，然后再摊平，乘于channel（embedding）长度

        #if num_edge_feats > 0:
        #    self.w_e2l = nn.Linear(num_edge_feats, num_node_feats)
        if self.output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)  #最后加个全连接

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        # initial parameter
        # if needed

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        # step 1: concatenate node_features and the node-relevant edge_features
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

        # step 2: propagate the message via 4 GCN layers
        out = self.conv1(x, edge_index)
        x = torch.cat([x, out], 1)
        out = self.conv2(out, edge_index)
        x = torch.cat([x, out], 1)
        out = self.conv3(out, edge_index)
        x = torch.cat([x, out], 1)
        out = self.conv4(out, edge_index)
        x = torch.cat([x, out], 1)

        # step 3: sortpooling layer
        
        

    def node_4edge_slice(self, edge_index):
        row, _ = edge_index[0], edge_index[1]
        row = row.cpu().numpy()
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(row)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])
        return node_slice


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN(1, datasets.num_node_features,
              datasets.num_edge_features).to(device)
data = datasets.data.to(device)
loader = DataLoader(datasets, batch_size=1, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
optimizer.zero_grad()
for batch in loader:
    batch.to(device)
    out = model(batch)
    break

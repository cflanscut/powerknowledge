from torch.functional import Tensor
from torch_geometric.data import InMemoryDataset
import os.path as osp
import torch
import torch.nn as nn
from torch_geometric.io import read_tu_data
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np


class PLAIDDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name='PLAIDG_single',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_edge_attr=False,
                 use_node_label=False):
        self.name = name
        super(PLAIDDataset, self).__init__(root, transform, pre_transform,
                                           pre_filter)  # 预处理在这里实现
        self.data, self.slices = torch.load(
            self.processed_paths[0])  # 读取处理好的数据
        if self.data.x is not None and not use_node_label:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, :num_node_attributes]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:,
                                                      num_edge_attributes:]  # 不需要edge_attr，只要后面的label

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')  # join可以自动加斜杠

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(
        self
    ):  #A list of files in the raw_dir which needs to be found in order to skip the download.
        names = [
            'A', 'edge_attributes', 'graph_indicator', 'graph_labels',
            'node_attributes', 'node_labels'
        ]
        return ['{}_{}.txt'.format('PLAIDG', name) for name in names]

    @property
    def processed_file_names(
        self
    ):  #A list of files in the processed_dir which needs to be found in order to skip the processing.
        return 'data.pt'

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    def download(self):
        print('This dataset is not yet public')

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, 'PLAIDG')

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]  #拆成一个个单图
            data_list = [self.pre_transform(data)
                         for data in data_list]  #对每个单图进行pre_transform
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))


class DGCNN(torch.nn.Module):
    def __init__(
        self,
        output_dim,
        num_node_feats,
        num_edge_feats,
        num_class,
        latent_dim=[2, 1],
        k=34,
        conv1d_channels=[1, 32],
        conv1d_kws=[0, 5],  # 0是待定的意思
        conv1d_activation='ReLU',
    ):
        super(DGCNN, self).__init__()

        self.latent_dim = latent_dim  # 指的是图卷积里面的每个层的节点特征维度，最后一层为1，论文写着方便排序
        self.output_dim = output_dim  # 指的是图卷积之后输出的维度，如果有dense则输出dense之后的，没有dense就输出dense
        self.num_node_feats = num_node_feats  # dataset.num_node_features
        self.num_edge_feats = num_edge_feats  # dataset.num_edge_features
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim  # 把所有图层拼接起来作为1-D卷积的输入
        self.hidden_dim = int(math.sqrt(output_dim * num_class))

        # GCN
        self.conv1 = GCNConv(self.num_node_feats, self.latent_dim[0])
        self.conv2 = GCNConv(self.latent_dim[0], self.latent_dim[1])
        # self.conv3 = GCNConv(self.latent_dim[1], self.latent_dim[2])
        # self.conv4 = GCNConv(self.latent_dim[2], self.latent_dim[3])

        # conv1d
        # self.conv1d_params1 = nn.Conv1d(
        #     1, conv1d_channels[0], conv1d_kws[0],
        #     conv1d_kws[0])  # 一维输入则通道为1，所有节点特征拼成1维了，核长等于步长，一步一个节点
        # # self.maxpool1d = nn.MaxPool1d(2, 2)  # 窗口大小2，步长2，沿着一维的方向(类似于两个节点的输出取最大)
        # # self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
        # #                                 conv1d_kws[1], 1)
        # # dense-layers
        # dense_dim = int(
        #     k)  # 这个跟池化的输出有关。如100个节点卷完第一层后变成100个数，再池化成50个节点（每个节点有embedding维度）
        # self.dense_dim = (dense_dim) * conv1d_channels[
        #     0]  # 一维的长度是池化后再一维卷，即dense-kws+1，然后再摊平，乘于channel（embedding）长度
        self.conv1dlist = nn.ModuleList([
            nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0]) for i in range(k)
        ])
        self.dense_dim = k
        if self.output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)  #最后加个全连接

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        # MLPClassifier
        # self.h1_weights = nn.Linear(output_dim, self.hidden_dim)
        # self.h2_weights = nn.Linear(self.hidden_dim, num_class)
        self.h1_weights = nn.Linear(output_dim, num_class)

        # initial parameter
        # if needed

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index

        graph_num = len(np.unique(
            data.batch))  # len(np.unique(batch.batch))为batch中几个图
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(data.batch)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])

        # step 1: extra node-feature and node-index
        # node_index = x[:, datasets.data.num_node_attributes:]
        # x = x[:, :datasets.data.num_node_attributes]

        # step 2: propagate the message via 4 GCN layers
        out = self.conv1(x, edge_index)
        # x = torch.cat([x, out], 1)
        x = out
        out = self.conv2(out, edge_index)
        x = torch.cat([x, out], 1)
        # out = self.conv3(out, edge_index)
        # x = torch.cat([x, out], 1)
        # out = self.conv4(out, edge_index)  # 最后一层GCN的输出，N*1维，作为sortpooling排序用
        # x = torch.cat([x, out], 1)

        # step 3: sortpooling layer
        # 因为batch里不止一张图，所以要区分每个图单独进行sortpooling
        # batch_sortpooling_graphs = torch.zeros(graph_num, self.k,
        #                                        self.total_latent_dim)
        # if torch.cuda.is_available() and isinstance(x, torch.cuda.FloatTensor):
        #     batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        # for i in range(graph_num):
        #     # to_sort = out[node_slice[i]:node_slice[i + 1]]  # 把当前graph的节点特征拿出来
        #     # k = self.k if self.k <= node_slice[i + 1] - node_slice[
        #     #     i] else node_slice[i + 1] - node_slice[i]  # 判断k和图节点数的大小关系
        #     # _, topk_indices = to_sort.topk(k, dim=0)  # 返回前k个元素及对应的索引
        #     topk_indices = Tensor(range(0, self.k))
        #     topk_indices += node_slice[i]  # 定位到原始batch的索引
        #     topk_indices = topk_indices.squeeze()
        #     topk_indices = topk_indices.int()
        #     sortpooling_graph = x.index_select(0, topk_indices)
        #     # if k < self.k:
        #     #     to_pad = torch.zeros(self.k - k, self.total_latent_dim)
        #     #     if torch.cuda.is_available() and isinstance(
        #     #             x, torch.cuda.FloatTensor):
        #     #         to_pad = to_pad.cuda()
        #     #     sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
        #     batch_sortpooling_graphs[i] = sortpooling_graph

        # step 4：traditional 1d convolution and dense layers
        # to_conv1d = batch_sortpooling_graphs.view(
        #     (-1, 1, self.k * self.total_latent_dim))  # 每个图变成1x1xk*totaldim的大小
        to_conv1d = x.view(-1, self.k, self.total_latent_dim)
        to_dense = torch.zeros(graph_num, self.k, 1)
        for i, m in enumerate(self.conv1dlist):
            # out = m(to_conv1d[:, i, :].view(-1, 1, self.total_latent_dim))
            to_dense[:, i, :] = m(to_conv1d[:, i, :].view(
                -1, 1, self.total_latent_dim)).view(-1, 1)
        to_dense = to_dense.squeeze()
        # conv1d_res = self.conv1d_params1(
        #     to_conv1d
        # )  # 一维卷积，一个图为一维向量，卷积核大小为单个节点特征长度，步长也为单个节点长度。输出长度为k，channel为16
        # conv1d_res = self.conv1d_activation(conv1d_res)  # ReLU
        # # conv1d_res = self.maxpool1d(conv1d_res)  # 窗口大小2，步长2，每个图为（k-2）/2+1
        # # conv1d_res = self.conv1d_params2(
        # #     conv1d_res
        # # )  # 继续卷，这时候卷积核大小为5，步长1，输出channel为32，即一个位置有32维的embedding。长度为((k-2)/2+1)-5+1
        # # conv1d_res = self.conv1d_activation(conv1d_res)  # ReLU

        # to_dense = to_dense.view(graph_num,
        #                          -1)  # 把channel拉平，变成G*上面的长度 x channel数

        if self.output_dim > 0:
            reluact_fp = self.out_params(to_dense)  # 如果有output_dim，加一个线性全连接
            # reluact_fp = self.conv1d_activation(out_linear)  # ReLU
        else:
            reluact_fp = to_dense  # 如果没有就直接输出

        # step 5: MLP classification
        logits = self.h1_weights(reluact_fp)
        # h1 = F.relu(h1)
        # h1 = F.dropout(h1, training=self.training)
        # logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        return logits

    def node_4edge_slice(self, edge_index):
        row, _ = edge_index[0], edge_index[1]
        row = row.cpu().numpy()
        node_slice4edge = torch.cumsum(torch.from_numpy(np.bincount(row)), 0)
        node_slice4edge = torch.cat([torch.tensor([0]), node_slice4edge])
        return node_slice4edge


path = osp.join(osp.abspath('.'), 'graph', 'PLAIDG_single')
datasets = PLAIDDataset(root=path,
                        name='PLAIDG_single',
                        use_node_label=False,
                        use_edge_attr=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DGCNN(20, datasets.num_node_features, datasets.num_edge_features,
              datasets.num_classes).to(device)
data = datasets.data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=5e-4)

model.train()
optimizer.zero_grad()
num_epochs = 100
for epoch in range(num_epochs):
    loader = DataLoader(datasets, batch_size=100, shuffle=True)
    total_loss = []
    for batch in loader:
        batch.to(device)
        out = model(batch)
        y = batch.y
        loss = F.nll_loss(out, y)
        pred = out.data.max(1, keepdim=True)[1]
        acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(
            y.size()[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        total_loss.append(np.array([loss, acc]) * len(y))

    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / len(datasets.data.y)
    print('epoch %d:loss %.5f acc %.5f' % (epoch, avg_loss[0], avg_loss[1]))
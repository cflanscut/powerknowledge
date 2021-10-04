from torch_geometric.data import InMemoryDataset
import os.path as osp
import torch
from torch_geometric.io import read_tu_data


class PLAIDDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 name='PLAIDG',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 use_edge_attr=False):
        self.name = name
        super(PLAIDDataset, self).__init__(root=root,
                                           transform=transform,
                                           pre_transform=pre_transform,
                                           pre_filter=pre_filter)  # 预处理在这里实现
        self.data, self.slices = torch.load(
            self.processed_paths[0])  # 读取处理好的数据
        if self.data.x is not None and not use_node_attr:  #若需要node_attr，则跳过这个判断
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:,
                                      num_node_attributes:]  # 不需要node_attr，只要后面的label
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
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(
        self
    ):  #A list of files in the processed_dir which needs to be found in order to skip the processing.
        return 'data.pt'

    def download(self):
        print('This dataset is not yet public')

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

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

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch
import numpy as np
from itertools import repeat, product

dataset = TUDataset(root='/home/chaofan/powerknowledge/graph/ENZYMES',
                    name='ENZYMES',
                    use_node_attr=True)
print(dataset.num_node_labels)
print(dataset.num_node_attributes)

# class people():
#     def __init__(self) -> None:
#         self.name = 'test'
#         if 'say_hello' in self.__class__.__dict__.keys():
#             self.say_hello()

#     def say_hello(self):
#         r"""say_hello."""
#         raise NotImplementedError

# class student(people):
#     def __init__(self) -> None:
#         super(student, self).__init__()
#         self.name = None

#     def say_hello(self):
#         r"""say_hello."""
#         raise NotImplementedError

# class master(student):
#     def __init__(self, name=None) -> None:
#         super(master, self).__init__()
#         self.name = name
#         print(self.name)

#     def say_hello(self):
#         print("Hello!")
#         print(self.name)

# lcf = master(name='lanchaofan')

# x = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
#                   [5, 6, 7]]).to(torch.long)
# edge_index = torch.tensor([[1, 1, 4], [2, 3, 5]]).to(torch.long)
# edge_attr = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).to(torch.long)
# y = torch.tensor([1, 2]).to(torch.long)
# batch = torch.tensor([1, 1, 1, 2, 2]).to(torch.long)
# data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
# print(data.keys)
# print(data.__class__())

# data1 = Data()
# data2 = Data()
# datalist = [data1, data2]
# i = 0
# for item, key in product(datalist, data.keys):
#     print(i)
#     print(item)
#     print(key)
#     i += 1
# print(data)
# data, slices = split(data, batch)
# print(data)
# print(slices)

# def split(data, batch):
#     node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)),
#                               0)  # slice类似于里程碑，即每个图的节点从第几号开始
#     node_slice = torch.cat([torch.tensor([0]), node_slice])  # 补0是因为graph从1开始编号

#     row, _ = data.edge_index
#     edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])),
#                               0)  # batch[row]得到每条边在哪个图
#     edge_slice = torch.cat([torch.tensor([0]),
#                             edge_slice])  # slice类似于里程碑，即每个图的边从第几号开始

#     # Edge indices should start at zero for every graph.
#     data.edge_index -= node_slice[batch[row]].unsqueeze(
#         0)  # 意思是每个图的边都从1开始编，不采用所有图的边一起编
#     data.__num_nodes__ = torch.bincount(batch).tolist()

#     slices = {'edge_index': edge_slice}
#     if data.x is not None:
#         slices['x'] = node_slice
#     if data.edge_attr is not None:
#         slices['edge_attr'] = edge_slice
#     if data.y is not None:
#         if data.y.size(0) == batch.size(0):  # 判断y是node标签还是graph标签
#             slices['y'] = node_slice
#         else:
#             slices['y'] = torch.arange(0, batch[-1] + 2,
#                                        dtype=torch.long)  #读取图数，直接第几个数就是第几个图

#     return data, slices  # slices是用来每个图遍历，找到对应的点和边在原始文件的位置

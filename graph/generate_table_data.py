'''
Author: LanChaofan cflanscut@163.com
Date: 2021-10-05 22:02:25
LastEditors: LanChaofan cflanscut@163.com
LastEditTime: 2022-09-26 20:51:24
Description: 

Copyright (c) 2022 by lanchaofan, All Rights Reserved.
'''
import os.path as osp
import numpy as np
file_dir = osp.join(osp.abspath(''), 'graph', 'PLAIDG_single', 'PLAIDG_single',
                    'raw')
with open(osp.join(file_dir,
                   'PLAIDG_graph_indicator.txt')) as PLAIDG_graph_indicator:
    graph_indicator = PLAIDG_graph_indicator.read().split('\n')[:-1]
node_slice = np.cumsum(np.bincount(graph_indicator), 0)
with open(osp.join(file_dir, 'PLAIDG_node_labels.txt')) as PLAIDG_node_labels:
    node_labels = PLAIDG_node_labels.read().split('\n')[:-1]
with open(osp.join(file_dir,
                   'PLAIDG_node_attributes.txt')) as PLAIDG_node_attributes:
    node_attributes = PLAIDG_node_attributes.read().split('\n')[:-1]
with open(osp.join(file_dir,
                   'PLAIDG_graph_labels.txt')) as PLAIDG_graph_labels:
    graph_labels = PLAIDG_graph_labels.read().split('\n')[:-1]
open(osp.join(file_dir, 'table_data.txt'), 'w').close()

count = 0
<<<<<<< HEAD
times_threshold = 9  # 20*0.3
=======
times_threshold = 8  # 20*0.4
>>>>>>> 667980873b1d3306a8e5668994fffdca0cfa5e43

for i in range(len(node_slice) - 1):
    node_list = {str(i): [-1] for i in range(30)}
    label = graph_labels[i]
    for j in range(node_slice[i], node_slice[i + 1]):
        feature = node_labels[j]
        times = int(node_attributes[j].split(',')[1])
        if times > times_threshold:
            node_list[feature].append(int(node_attributes[j].split(',')[0]))

    with open(osp.join(file_dir, 'table_data.txt'), 'a') as table_data:
        single_idx = []
        multi_idx = []
        total_len = 1
        for k, v in node_list.items():  # 有多个特征的，去掉-1项
            if len(v) > 1:
                node_list[k] = v[1:]
                total_len *= (len(v) - 1)
            if len(v) == 2:
                single_idx.append(k)
            # elif len(v) >= 2:
            else:  # -1的放这里
                multi_idx.append(k)
        new_data = np.zeros((total_len, len(node_list.keys()) + 1))
        count += total_len
        for f in single_idx:
            new_data[:, int(f)] = node_list[f]
        base_times = 1
        for f in multi_idx:
            f_len = len(node_list[f])
            start_times = base_times
            base_times *= f_len
            for idx, v in enumerate(node_list[f]):
                for x in range(idx * start_times, total_len, base_times):
                    new_data[x:x + start_times, int(f)] = v
        new_data[:, -1] = graph_labels[i]
        # if i == 513:
        #     continue
        for line in new_data:
            for value in line:
                table_data.write(str(value) + ',')
            table_data.write('\n')
    print('{}/{} total_len:{} count:{}'.format(i + 1, len(graph_labels),
                                               total_len, count))
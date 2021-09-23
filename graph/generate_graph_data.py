# %%
# 导入模块
import time
import sys
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from itertools import product
# %%
# 读取数据
sys.path.append('/home/chaofan/powerknowledge/data')
from read_PLAID_data import read_processed_data

start_reading_time = time.time()

feature_select = [
    'i_mean', 'i_wave_factor', 'i_pp_rms', 'i_thd', 'pure_thd', 'P', 'Q',
    'P_F', 'i_hp1', 'z_hp1', 'i_hm2', 'z_hm2', 'i_hp2', 'z_hp2', 'i_hm3',
    'z_hm3', 'i_hp3', 'z_hp3', 'i_hm4', 'z_hm4', 'i_hp4', 'z_hp4', 'i_hm5',
    'z_hm5', 'i_hp5', 'z_hp5', 'i_hm6', 'z_hm6', 'i_hp6', 'z_hp6', 'i_hm7',
    'z_hm7', 'i_hp7', 'z_hp7'
]  # 选择所用特征量

selected_label = [
    'Air Conditioner', 'Blender', 'Coffee maker', 'Fan', 'Fridge', 'Hair Iron',
    'Hairdryer', 'Heater', 'Incandescent Light Bulb', 'Microwave',
    'Soldering Iron', 'Vacuum', 'Washing Machine', 'Water kettle'
]  # 选择所用电器

load_transformer = {}
num = 0
for item in selected_label:
    load_transformer[item] = num
    num += 1

each_file_len = 20
# selected_label = ['Air Conditioner']

x_train, y_train, index_train = read_processed_data(
    'type',
    type_header='appliance',
    selected_label=selected_label,
    direaction=1,
    offset=0,
    each_lenth=each_file_len,
    feature_select=feature_select,
    Transformer=load_transformer,
    source='submetered_process2.1/training')  # 读取训练数据
y_train = y_train.reshape(-1, 1)

x_validation, y_validation, index_validation = read_processed_data(
    'type',
    type_header='appliance',
    selected_label=selected_label,
    direaction=1,
    offset=0,
    each_lenth=each_file_len,
    feature_select=feature_select,
    Transformer=load_transformer,
    source='submetered_process2.1/validation')  # 读取验证数据
y_validation = y_validation.reshape(-1, 1)

x_trainval = np.concatenate((x_train, x_validation), axis=0)
y_trainval = np.concatenate((y_train, y_validation), axis=0)

x_test, y_test, index_test = read_processed_data(
    'type',
    type_header='appliance',
    selected_label=selected_label,
    direaction=1,
    offset=0,
    each_lenth=each_file_len,
    feature_select=feature_select,
    Transformer=load_transformer,
    source='submetered_process2.1/testing')  # 读取测试数据

print('finished loading data, cost %.3fs' % (time.time() - start_reading_time))

# %%
# 特征值聚类
center_record = {}
centers = {}
feature_lens = 30  # 应该改成len函数取值
for f_index in range(feature_lens):  #
    # n_cluster = 4 * (2 * feature_lens - 2 * f_index) - 4 # 按照数圈的方式制定聚类中心
    n_cluster = 10
    km = KMeans(n_clusters=n_cluster, random_state=1)
    if f_index in [5, 6]:  #只有这两个特征（功率）需要取对数
        x = np.log(x_train[:, f_index]).reshape(-1, 1)
    x = x_train[:, f_index].reshape(-1, 1)
    y_pred = km.fit_predict(x).reshape(-1,
                                       1)  # 这里的y只是kmean里面的聚类中心编号，不是从小到大排列下来的编号
    y_train = np.concatenate((y_train, y_pred), axis=1)
    centers[f_index] = [i for item in km.cluster_centers_
                        for i in item]  # item是个数组，如果只有1维，再加个循环读取数值
    center_record[f_index] = np.argsort(
        centers[f_index]
    )  # 对kmeans得到的聚类中心从小到大进行排序，得到排序index，即center[record[0]]为中心最小值
y_label = y_train[:, 0]
y_cluster = y_train[:, 1:]
node_num_transform = {}
node_label_transform = {}
tmp = 1  # 好像公开数据集节点都是从1开始的
for feat, cluster in product(range(feature_lens),
                             range(n_cluster)):  #把所有特征的聚类值转化为顺序排列的整数
    node_num_transform['%03d,%03d' % (feat, cluster)] = str(
        tmp)  #转为用字符串，方便后续存储
    node_label_transform[tmp] = np.array([feat, cluster
                                          ])  # 暂时用不到拆分的，直接用tmp来代表特征和聚类中心
    tmp += 1
del tmp

# %%
# 计算图特征并存储成数据


# 输入点列表，返回所有边
def calc_edge(Node_list):
    edge_list = []
    for node_A in Node_list:
        for node_B in Node_list:
            if node_B == node_A:
                continue
            edge_list.append(node_A + ',' + node_B)
    return edge_list


# 记录边的重复出现次数
def count_edge(edge_count_dict, edge_list):
    for edge in edge_list:
        if edge not in edge_count_dict.keys():
            edge_count_dict[edge] = 1
        else:
            edge_count_dict[edge] += 1
    return edge_count_dict  # key为边，格式为'%03d,%03d'；value为出现的次数


# 记录点的重复出现次数
def count_node(node_count_dict, node_list):  # node为字符串型
    for node in node_list:
        if node not in node_count_dict.keys():
            node_count_dict[node] = 1
        else:
            node_count_dict[node] += 1
    return node_count_dict


def save_graph_in_txt(node_counts, edge_counts, graph_label) -> None:

    with open(
            "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_node_labels.txt",
            'r') as file_node_labels:
        base_node_num = len(file_node_labels.readlines())
    with open(
            "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_graph_label.txt",
            'r') as file_graph_label:
        base_graph_num = len(file_graph_label.readlines())
    with open(
            "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_graph_label.txt",
            'a') as file_graph_label:
        file_graph_label.write(str(graph_label) + '\n')

    tmp = 1
    index_convert = {}
    for key in node_counts.keys():  # key为节点类型，变成label；value为次数，变成attribute
        with open(
                "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_node_labels.txt",
                'a') as file_node_labels:
            file_node_labels.write(str(key) + '\n')  #指示当前节点的标签（特征+中心所对应的编号）
        with open(
                "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_node_attributes.txt",
                'a') as file_node_attributes:
            file_node_attributes.write(str(node_counts[key]) +
                                       '\n')  #指示当前节点的特征
        with open(
                "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_graph_indicator.txt",
                'a') as file_graph_indicator:
            file_graph_indicator.write(str(base_graph_num + 1) +
                                       '\n')  #指示当前节点属于第几个graph
        index_convert[key] = str(base_node_num + tmp)  #用于描述A矩阵，重新定义边的节点编号
        tmp += 1
    for key in edge_counts.keys():  # key为节点对，变成A阵；value为次数，变成attribute
        key_str = key.split(',')
        node_a = index_convert[key_str[0]]
        node_b = index_convert[key_str[1]]
        with open("/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_A.txt",
                  'a') as file_A:
            file_A.write(node_a + ',' + node_b + '\n')
        with open(
                "/home/chaofan/powerknowledge/graph/PLAIDG/PLAIDG_edge_attributes.txt",
                'a') as file_edge_attributes:
            file_edge_attributes.write(str(edge_counts[key]) + '\n')


file_count = 1
for i, yi_cluster in enumerate(y_cluster):
    # if i < 100:
    #     continue  # 跳过100个数？
    if i % each_file_len == 0:  #判断是否进入新的文件
        edge_counts = {}
        node_counts = {}
        y_label_tmp = y_label[i]
    nodes_temp = []

    for j, cluster_k in enumerate(yi_cluster):  # j为特征编号,cluster_k为所属类别
        cluster_k = int(cluster_k)
        nodes_temp.append(node_num_transform["%03d,%03d" %
                                             (j, cluster_k)])  # 读取当前特征的节点编号

    # calc edge
    edge_list = calc_edge(nodes_temp)
    edge_counts = count_edge(edge_counts, edge_list)
    # calc node
    node_counts = count_node(node_counts, nodes_temp)
    if i % each_file_len == 19:
        save_graph_in_txt(node_counts, edge_counts, y_label_tmp)
        print('%04d/%04d' % (file_count, int(len(y_label) / each_file_len)))
        file_count += 1

# %%

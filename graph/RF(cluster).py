# %%
# 导入模块
import time
import sys
import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans
from collections import Counter
from itertools import product
import os.path as osp
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import shap
# %%
# 读取数据
sys.path.append('.')
from data.read_PLAID_data import read_processed_data, get_feature_name

start_reading_time = time.time()
source_file_pack = 'submetered_zengj'
# feature_select = [
#     'i_mean', 'i_wave_factor', 'i_pp_rms', 'i_thd', 'pure_thd', 'P', 'Q',
#     'P_F', 'i_hp1', 'z_hp1', 'i_hm2', 'z_hm2', 'i_hp2', 'z_hp2', 'i_hm3',
#     'z_hm3', 'i_hp3', 'z_hp3', 'i_hm4', 'z_hm4', 'i_hp4', 'z_hp4', 'i_hm5',
#     'z_hm5', 'i_hp5', 'z_hp5', 'i_hm6', 'z_hm6', 'i_hp6', 'z_hp6', 'i_hm7',
#     'z_hm7', 'i_hp7', 'z_hp7'
# ]  # 选择所用特征量

feature_select = get_feature_name(dir=osp.join(source_file_pack, 'total'))

selected_label = [
    'Air Conditioner', 'Blender', 'Coffee maker', 'Compact Fluorescent Lamp',
    'Fan', 'Fridge', 'Hair Iron', 'Hairdryer', 'Heater',
    'Incandescent Light Bulb', 'Laptop', 'Microwave', 'Soldering Iron',
    'Vacuum', 'Washing Machine', 'Water kettle'
]  # 选择所用电器

load_transformer = {}
num = 0
for item in selected_label:
    load_transformer[item] = num
    num += 1

each_file_len = 20

x_train, y_train, index_train = read_processed_data(
    'type',
    type_header='appliance',
    selected_label=selected_label,
    direaction=1,
    offset=0,
    each_lenth=each_file_len,
    feature_select=feature_select,
    Transformer=load_transformer,
    source=osp.join(source_file_pack, 'training'))  # 读取训练数据
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
    source=osp.join(source_file_pack, 'validation'))  # 读取验证数据
y_validation = y_validation.reshape(-1, 1)

x_trainval = np.concatenate((x_train, x_validation), axis=0)
y_trainval = np.concatenate((y_train, y_validation), axis=0)
y_trainval = y_trainval.ravel()

x_test, y_test, index_test = read_processed_data('type',
                                                 type_header='appliance',
                                                 selected_label=selected_label,
                                                 direaction=1,
                                                 offset=0,
                                                 each_lenth=each_file_len,
                                                 feature_select=feature_select,
                                                 Transformer=load_transformer,
                                                 source=osp.join(
                                                     source_file_pack,
                                                     'testing'))  # 读取测试数据
y_test = y_test.ravel()
print('finished loading data, cost %.3fs' % (time.time() - start_reading_time))

# %%
# 原始数据进行随机深林
rf_base = RandomForestClassifier()
rf_base.fit(x_trainval, y_trainval)
y_test_pred = rf_base.predict(x_test)
print("Original Accuracy : %.4g" % metrics.accuracy_score(y_test, y_test_pred))
explainer = shap.TreeExplainer(rf_base)
shap_values = explainer.shap_values(x_trainval)
shap.summary_plot(shap_values, feature_select, plot_type='bar', max_display=30)

%%
特征值聚类
center_record = {}
centers = {}
feature_lens = 30  # 应该改成len函数取值

for n_cluster in range(1, 20):
    data_train = y_trainval.reshape(-1, 1)
    data_test = y_test.reshape(-1, 1)
    for f_index in range(feature_lens):  #
        # n_cluster = 4 * (2 * feature_lens - 2 * f_index) - 4 # 按照数圈的方式制定聚类中心
        # n_cluster = 20
        km = KMeans(n_clusters=n_cluster, random_state=1)
        if f_index in [5, 6]:  # 只有这两个特征（功率）需要取对数
            x_trainval_f = np.log(x_trainval[:, f_index]).reshape(-1, 1)
            x_test_f = np.log(x_test[:, f_index]).reshape(-1, 1)
        x_trainval_f = x_trainval[:, f_index].reshape(-1, 1)
        x_test_f = x_test[:, f_index].reshape(-1, 1)
        # 对训练集进行聚类
        x_trainval_cluster = km.fit_predict(x_trainval_f).reshape(
            -1, 1)  # 这里的y只是kmean里面的聚类中心编号，不是从小到大排列下来的编号
        data_train = np.concatenate((data_train, x_trainval_cluster), axis=1)
        # 对测试集进行聚类转化
        x_test_cluster = km.predict(x_test_f).reshape(-1, 1)  # 直接预测就行，不需要fit
        data_test = np.concatenate((data_test, x_test_cluster), axis=1)
        # 记录聚类中心和排序
        centers[f_index] = [i for item in km.cluster_centers_
                            for i in item]  # item是个数组，如果只有1维，再加个循环读取数值
        center_record[f_index] = np.argsort(
            centers[f_index]
        )  # 对kmeans得到的聚类中心从小到大进行排序，得到排序index，即center[record[0]]为中心最小值

    y_train = data_train[:, 0]
    x_train = data_train[:, 1:]
    y_test = data_test[:, 0]
    x_test = data_test[:, 1:]

    rf_cluster = RandomForestClassifier()
    rf_cluster.fit(data_train[:, 1:], data_train[:, 0])
    y_test_pred = rf_cluster.predict(data_test[:, 1:])
    print("%02d: Cluster Accuracy : %.4g" %
          (n_cluster, metrics.accuracy_score(data_test[:, 0], y_test_pred)))

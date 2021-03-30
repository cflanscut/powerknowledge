import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def read_index(type, header='appliance'):
    """
    读取给定type所包含的样本序号。如：读取load里面'R','I','NL'对应的设备序号有哪些
    :param type: 需要查询的标签名
    :param header: 对应的json里面的键，默认appliance
    """
    with open(
            '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
            'r',
            encoding='utf8') as load_meta:
        meta = json.load(load_meta)
        metadata_len = len(meta)
        label_index = {}
        for i in range(metadata_len):
            try:
                label = meta[str(i + 1)][header][str(type)]
            except NameError:
                print('第{}个样本没有属性{}'.format(i + 1, type))
            else:
                if label in label_index.keys():
                    label_index[label].append(i + 1)
                else:
                    label_index[label] = [i + 1]
        return label_index


# test read_index
# ri = read_index('type', header="appliance")
# print(ri)


def read_correlation(name,
                     type,
                     name_header='appliance',
                     type_header='appliance'):

    with open(
            '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
            'r',
            encoding='utf8') as load_meta:
        meta = json.load(load_meta)
        metadata_len = len(meta)
        name_type = {}
        for i in range(metadata_len):
            try:
                name_label = meta[str(i + 1)][name_header][str(name)]
                type_label = meta[str(i + 1)][type_header][str(type)]
            except NameError:
                print('第{}个样本没有属性{}或{}'.format(i + 1, name, type))
            else:
                if name_label in name_type.keys():
                    if type_label in name_type[name_label]:
                        continue
                    else:
                        name_type[name_label].append(type_label)
                else:
                    name_type[name_label] = []
                    name_type[name_label].append(type_label)
        return name_type


# test
# rc = read_correlation('type', 'brand')
# print(rc)


def read_processed_data(type,
                        type_header='appliance',
                        selected_label=[],
                        offset=0,
                        direaction=0,
                        each_lenth=1,
                        feature_select=None,
                        Transformer=None,
                        source='submetered_process'):
    meta = None
    with open(
            '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
            'r',
            encoding='utf8') as load_meta:
        meta = json.load(load_meta)
    dir = '/home/chaofan/powerknowledge/data/source/' + source
    csv_list = os.listdir(dir)
    first_data = pd.read_csv(os.path.join(dir, csv_list[0]))
    features = first_data.keys()
    feature_len = 0
    if feature_select is not None:
        try:
            feature_index = features.get_indexer(feature_select)
            feature_index = feature_index.tolist()
        except IndexError:
            print('there is no feature-selected in data')
    if feature_select is None:
        feature_len = len(features)
    else:
        try:
            feature_len = len(feature_select)
        except TypeError:
            print('feature_select需为所选特征数组')
    x = np.zeros((len(csv_list) * each_lenth, feature_len))
    y = np.zeros((len(csv_list) * each_lenth))
    index = np.zeros((len(csv_list) * each_lenth, 2))
    for i, file in enumerate(csv_list):
        if feature_select is None:
            data = pd.read_csv(os.path.join(dir, file))
        else:
            data = pd.read_csv(
                os.path.join(dir, file),
                usecols=feature_index,
            )
        num = file[0:-4]
        data_len = len(data)
        try:
            label = meta[num][type_header][type]
        except TypeError:
            print('没有该属性')
        if selected_label != []:
            if label not in selected_label:
                continue
        if direaction == 0:
            for j in range(each_lenth):
                x[i * each_lenth + j, :] = data.loc[offset + j]
                index[i * each_lenth + j, 0] = num
                index[i * each_lenth + j, 1] = offset + j
                if Transformer is not None:
                    y[i * each_lenth + j] = Transformer[label]
                else:
                    y = y.astype(np.str)
                    y[i * each_lenth + j] = label
        else:
            for j in range(each_lenth):
                x[i * each_lenth + j, :] = data.loc[data_len - offset - j - 1]
                index[i * each_lenth + j, 0] = num
                index[i * each_lenth + j, 1] = data_len - offset - j - 1
                if Transformer is not None:
                    y[i * each_lenth + j] = Transformer[label]
                else:
                    y = y.astype(np.str)
                    y[i * each_lenth + j] = label
    idx = np.argwhere(np.all(x[:, ...] == 0, axis=1))
    x = np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    index = np.delete(index, idx, axis=0)
    x = np.nan_to_num(x)
    x[x > 10000] = 10000
    return x, y, index


def get_feature_name():
    dir = '/home/chaofan/powerknowledge/data/source/submetered_process2'
    csv_list = os.listdir(dir)
    first_file = pd.read_csv(os.path.join(dir, csv_list[0]))
    features = first_file.keys()
    features = list(features)
    del features[0]
    return features


def read_source_data(file_dir, length='default', offset=0):
    soucre_data = pd.read_csv(file_dir, names=["I", "U"])
    if length == 'default':
        length = soucre_data.shape[0] - offset - 1
    return soucre_data['U'][offset:offset + length -
                            1], soucre_data['I'][offset:offset + length - 1]


def find_temp_start(guide_feature, threshold):
    dir = '/home/chaofan/powerknowledge/data/source/submetered_process2.1'
    start_record = {}
    csv_list = os.listdir(dir)
    for i, file in enumerate(csv_list):
        if '.csv' not in file:
            continue
        value0 = 0
        data = pd.read_csv(os.path.join(dir, file))
        data = data[guide_feature]
        for j, value in enumerate(data):
            if abs(value - value0) >= threshold:
                start_record[file] = j
                break
            else:
                value0 = value
    return start_record


# # test
# t = find_temp_start('P', 5)
def find_temp_start_pricisely(guide_feature, threshold):
    source_dir = 'data/source/submetered_new/'
    start_index = find_temp_start(guide_feature, threshold)
    start_record = {}
    for file, start_period in start_index.items():
        if start_period > 0:
            start_period -= 1
            file_dir = source_dir + file
            Voltage, Current = read_source_data(file_dir,
                                                offset=start_period * 500,
                                                length=1000)
            start_i = 0
            value0 = 0
            consecutive_count = 0
            sign = 0
            for i, value in enumerate(Current):
                diff = value - value0
                if np.sign(diff) == sign and sign != 0:
                    if consecutive_count == 0:
                        start_i = start_period * 500 + i
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                if consecutive_count > 5:
                    start_record[file] = start_i
                    break
                value0 = value
                sign = np.sign(diff)
    return start_record


# test
# t = find_temp_start_pricisely('P', 5)
def bar(pos):
    pos = int(pos)
    bar.ax.clear()
    if pos + bar.N > len(bar.x):
        n = len(bar.x) - pos
    else:
        n = bar.N
    X = bar.x[pos:pos + n]
    Y = bar.y[pos:pos + n]
    bar.ax.plot(X, Y)  # 相当于每次触发重新画

    bar.ax.xaxis.set_ticks([])
    bar.ax.yaxis.set_ticks([])


def slider_plot(data):
    fig, bar.ax = plt.subplots(figsize=(10, 6))
    bar.x = np.linspace(0, len(data) - 1, len(data))
    bar.y = np.array(data)
    bar.N = 3000
    barpos = plt.axes([0.18, 0.05, 0.55, 0.03],
                      facecolor="skyblue")  # [左下角在画板的位置（横坐标纵坐标），图大小（长宽）]
    slider = Slider(barpos, 'Barpos', 0, len(bar.x) - bar.N,
                    valinit=0)  # barpos放置滑条容器，用axes实例；0最小值；后面接着最大值；valinit初始值
    slider.on_changed(
        bar)  # on_changed是滑动条方法，用于绑定滑动条值改变的事件，就是滑动条改变了就调用这个函数，会自动传滑动条当前值进去
    bar(0)
    plt.show()
    plt.close()
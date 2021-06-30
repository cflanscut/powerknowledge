import pandas as pd
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def read_source_data(file_dir, length='default', offset=0):
    soucre_data = pd.read_csv(file_dir,
                              names=["U", "I"],
                              skiprows=9,
                              usecols=[1, 2])
    if length == 'default':
        length = soucre_data.shape[0] - offset - 1
    return soucre_data['U'][offset:offset + length -
                            1], soucre_data['I'][offset:offset + length - 1]


# test
# Temp_U, Temp_I = read_source_data(
#     'data/source/HIOKI/source/DDCsx_20201126_170239.csv', length=800)
# print()


def read_processed_data(
    type,
    type_header='appliance',
    selected_label=[],
    offset=0,
    direaction=0,
    each_lenth=1,
    feature_select=None,
    Transformer=None,
    source='/HIOKI_xinpei/process/total',
    source_json='/home/chaofan/powerknowledge/data/source/HIOKI_xinpei/label.json'
):
    meta = None
    with open(
            # '/home/chaofan/powerknowledge/data/source/metadata_submetered2.1.json',
            source_json,
            'r',
            encoding='utf8') as load_meta:
        meta = json.load(load_meta)
    dir = '/home/chaofan/powerknowledge/data/source' + source
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
    for i, file in enumerate(csv_list):
        if feature_select is None:
            data = pd.read_csv(os.path.join(dir, file))
        else:
            data = pd.read_csv(
                os.path.join(dir, file),
                usecols=feature_index,
            )
        fragment = file.split('_')[:-2]
        for part, txt in enumerate(fragment):
            if part == 0:
                key_name = txt
            else:
                key_name = key_name + '_' + txt
        data_len = len(data)
        try:
            label = meta[key_name][type_header][type]
        except TypeError:
            print('没有该属性')
        if selected_label != []:
            if label not in selected_label:
                continue

        if direaction == 0:
            for j in range(each_lenth):
                x[i * each_lenth + j, :] = data.loc[offset + j]

                if Transformer is not None:
                    y[i * each_lenth + j] = Transformer[label]
                else:
                    y = y.astype(np.str)
                    y[i * each_lenth + j] = label
        else:
            for j in range(each_lenth):
                x[i * each_lenth + j, :] = data.loc[data_len - offset - j - 1]

                if Transformer is not None:
                    y[i * each_lenth + j] = Transformer[label]
                else:
                    y = y.astype(np.str)
                    y[i * each_lenth + j] = label
    idx = np.argwhere(np.all(x[:, ...] == 0, axis=1))
    x = np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    x = np.nan_to_num(x)
    x[x > 10000] = 10000
    x = np.delete(x, [0], axis=1)
    return x, y


# test
selected_label = ['Fan']
test_x, test_y = read_processed_data('type',
                                     selected_label=selected_label,
                                     direaction=1,
                                     each_lenth=1)


def find_temp_start(guide_feature, threshold):
    dir = '/home/chaofan/powerknowledge/data/source/HIOKI/process/'
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


# test
# t = find_temp_start('P', 5)
# print()


def find_temp_start_pricisely(guide_feature, threshold):
    source_dir = '/home/chaofan/powerknowledge/data/source/HIOKI/source/'
    start_index = find_temp_start(guide_feature, threshold)
    start_record = {}
    for file, start_period in start_index.items():
        if start_period > 0:
            start_period -= 1
            file_dir = source_dir + file
            Voltage, Current = read_source_data(file_dir,
                                                offset=start_period * 400,
                                                length=1000)
            start_i = 0
            value0 = 0
            consecutive_count = 0
            sign = 0
            for i, value in enumerate(Current):
                diff = value - value0
                if np.sign(diff) == sign and sign != 0:
                    if consecutive_count == 0:
                        start_i = start_period * 400 + i
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                if consecutive_count > 5:
                    if start_i > 100:
                        start_i -= 100
                    start_record[file] = start_i
                    break
                value0 = value
                sign = np.sign(diff)
    return start_record


# test
# source_dir = 'data/source/HIOKI/source/'
# new_dir = 'data/source/HIOKI/source_pured/'
# start_index = find_temp_start_pricisely('P', 5)
# count = 0

# for file, start_row in start_index.items():
#     count += 1
#     df = pd.read_csv(source_dir + file, skiprows=start_row)
#     df.to_csv(new_dir + file, index=False)
#     print('dealing...:%03d/%03d' % (count, len(start_index)))


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
    # bar.ax.yaxis.set_ticks([])


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


# Temp_U, Temp_I = read_source_data(
#     'data/source/HIOKI/source_pured/DDClyk_20210402_111655.csv', length=4000)
# slider_plot(Temp_I)


def find_stable_start(guide_feature,
                      threshold=5,
                      consecutive_count=3,
                      stable_rate=0.05):
    dir = '/home/chaofan/powerknowledge/data/source/HIOKI/process/'
    start_record = {}
    csv_list = os.listdir(dir)
    for i, file in enumerate(csv_list):
        if '.csv' not in file:
            continue
        data = pd.read_csv(os.path.join(dir, file))
        data = data[guide_feature]
        value0 = 0
        count = 0
        start_record[file] = []
        threshold = 5
        for j, value in enumerate(data):
            if abs(value / (value0 + 0.0001) - 1) <= stable_rate:
                if abs(value) < threshold:
                    value0 = value
                    continue
                count += 1
                value0 = value
                if count >= consecutive_count:
                    start_record[file].append(j)
                    threshold = abs(value0 * 1.1)
                    count = 0
            else:
                count = 0
                value0 = value
    return start_record


# record = find_stable_start('P', 5)
# print()

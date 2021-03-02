import json
import os
import numpy as np
import pandas as pd


def read_index(type, header='appliance'):
    with open('data/source/metadata_submetered.json', 'r',
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
# ri = read_index('load')
# print(ri)


def read_correlation(name,
                     type,
                     name_header='appliance',
                     type_header='appliance'):
    with open('data/source/metadata_submetered.json', 'r',
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
                    name_type[name_label] = type_label
        return name_type


# # test read_correlation
# rc = read_correlation('type', 'load')
# print(rc)


def read_processed_data(type,
                        offset=0,
                        direaction=0,
                        each_lenth=1,
                        feature_select=None,
                        Transformer=None):
    meta = None
    with open('data/source/metadata_submetered.json', 'r',
              encoding='utf8') as load_meta:
        meta = json.load(load_meta)
    dir = 'data/source/submetered_process'
    csv_list = os.listdir(dir)
    first_data = pd.read_csv(os.path.join(dir, csv_list[0]))
    features = first_data.keys()
    feature_len = 0
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
        num = file[0:-4]
        data_len = len(data)
        try:
            label = meta[num]['appliance'][type]
        except TypeError:
            print('没有该属性')
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

    return x, y


# test
# x, y = read_processed_data('load')
# y_onehot = pd.get_dummies(y)
# y_onehot.head()
# print(y_onehot)
# print(pd.value_counts(y, sort=False))

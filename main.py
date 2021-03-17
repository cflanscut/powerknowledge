#!/home/chaofan/anaconda3/bin/python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/03/07 16:32:12
@Author      :chaofan
@version      :1.0
'''

import sys
import time
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import joblib
sys.path.append('data/')
sys.path.append('data/source/')
sys.path.append('data/model/deep_learning_model')
sys.path.append('data/model/knowledge_model')
from read_PLAID_data import read_processed_data

start_reading_time = time.time()
x_is_cool, y_is_cool = read_processed_data('is_cool',
                                           type_header='extra label',
                                           direaction=1,
                                           offset=30)
x_is_cool = x_is_cool[:, 1:]

x_is_heat, y_is_heat = read_processed_data('is_heat',
                                           type_header='extra label',
                                           direaction=1,
                                           offset=30)
x_is_heat = x_is_heat[:, 1:]

x_is_light, y_is_light = read_processed_data('is_light',
                                             type_header='extra label',
                                             direaction=1,
                                             offset=30)
x_is_light = x_is_light[:, 1:]

x_is_rotate, y_is_rotate = read_processed_data('is_rotate',
                                               type_header='extra label',
                                               direaction=1,
                                               offset=30)
x_is_rotate = x_is_rotate[:, 1:]

load_transformer = {'I': 0, 'R': 1, 'NL': 0}
x_load, y_load = read_processed_data('load',
                                     direaction=1,
                                     offset=30,
                                     Transformer=load_transformer)
x_load = x_load[:, 1:]
print('finished loading data, cost %.3fs' % (time.time() - start_reading_time))

start_train_time = time.time()
x_is_cool_train, x_is_cool_test, y_is_cool_train, y_is_cool_test = train_test_split(
    x_is_cool, y_is_cool, test_size=0.5, random_state=0)
start_tree_time = time.time()
cooler_clf = tree.DecisionTreeClassifier(max_depth=3, min_samples_split=100)
cooler_clf.fit(x_is_cool_train, y_is_cool_train)

x_is_heat_train, x_is_heat_test, y_is_heat_train, y_is_heat_test = train_test_split(
    x_is_heat, y_is_heat, test_size=0.5, random_state=0)
start_tree_time = time.time()
heater_clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=50)
heater_clf.fit(x_is_heat_train, y_is_heat_train)

x_is_light_train, x_is_light_test, y_is_light_train, y_is_light_test = train_test_split(
    x_is_light, y_is_light, test_size=0.5, random_state=0)
start_tree_time = time.time()
lighter_clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=10)
lighter_clf.fit(x_is_light_train, y_is_light_train)

x_is_rotate_train, x_is_rotate_test, y_is_rotate_train, y_is_rotate_test = train_test_split(
    x_is_rotate, y_is_rotate, test_size=0.5, random_state=0)
start_tree_time = time.time()
rotater_clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=25)
rotater_clf.fit(x_is_rotate_train, y_is_rotate_train)

x_load_train, x_load_test, y_load_train, y_load_test = train_test_split(
    x_load, y_load, test_size=0.5, random_state=0)
start_tree_time = time.time()
load_clf = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=25)
load_clf.fit(x_load_train, y_load_train)
print('finished training, cost %.3fs' % (time.time() - start_train_time))

start_time = time.time()
print('start reading data')
x_light, y_light = read_processed_data('type',
                                       selected_label=[
                                           'Compact Fluorescent Lamp',
                                           'Incandescent Light Bulb', 'Laptop'
                                       ],
                                       direaction=1,
                                       offset=30,
                                       each_lenth=1)
x_light = x_light[:, 1:]
x_cold, y_cold = read_processed_data(
    'type',
    selected_label=['Fridge', 'Air Conditioner'],
    direaction=1,
    offset=30,
    each_lenth=1)
x_cold = x_cold[:, 1:]
x_dislight_R, y_dislight_R = read_processed_data(
    'type',
    selected_label=['Hairdryer', 'Heater', 'Coffee maker', 'Water kettle'],
    direaction=1,
    offset=30,
    each_lenth=1)
x_dislight_R = x_dislight_R[:, 1:]
x_heat_I, y_heat_I = read_processed_data(
    'type',
    selected_label=['Microwave', 'Hair Iron', 'Soldering Iron'],
    direaction=1,
    offset=30,
    each_lenth=1)
x_heat_I = x_heat_I[:, 1:]
x_rotate_I, y_rotate_I = read_processed_data('type',
                                             selected_label=[
                                                 'Air Conditioner', 'Vacuum',
                                                 'Fan', 'Washing Machine',
                                                 'Blender'
                                             ],
                                             direaction=1,
                                             offset=30,
                                             each_lenth=1)
x_rotate_I = x_rotate_I[:, 1:]
print('finished loading data, cost %2.2fs' % (time.time() - start_time))

x_light_train, x_light_test, y_light_train, y_light_test = train_test_split(
    x_light, y_light, test_size=0.5)
x_cold_train, x_cold_test, y_cold_train, y_cold_test = train_test_split(
    x_cold, y_cold, test_size=0.5)
x_dislight_R_train, x_dislight_R_test, y_dislight_R_train, y_dislight_R_test = train_test_split(
    x_dislight_R, y_dislight_R, test_size=0.5)
x_heat_I_train, x_heat_I_test, y_heat_I_train, y_heat_I_test = train_test_split(
    x_heat_I, y_heat_I, test_size=0.5)
x_rotate_I_train, x_rotate_I_test, y_rotate_I_train, y_rotate_I_test = train_test_split(
    x_rotate_I, y_rotate_I, test_size=0.5)

light_tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
cold_tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
dislight_R_tree = tree.DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
heat_I_tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=2)
rotate_I_tree = tree.DecisionTreeClassifier(max_depth=6, min_samples_leaf=2)
light_tree.fit(x_light_train, y_light_train)
cold_tree.fit(x_cold_train, y_cold_train)
dislight_R_tree.fit(x_dislight_R_train, y_dislight_R_train)
heat_I_tree.fit(x_heat_I_train, y_heat_I_train)
rotate_I_tree.fit(x_rotate_I_train, y_rotate_I_train)

x_test = np.concatenate((x_light_test, x_cold_test, x_dislight_R_test,
                         x_heat_I_test, x_rotate_I_test),
                        axis=0)
y_test = np.concatenate((y_light_test, y_cold_test, y_dislight_R_test,
                         y_heat_I_test, y_rotate_I_test),
                        axis=0)


def classify_model(inputs):
    outputs = []
    tree_outputs = []
    for x in inputs:
        y = []
        j = []
        x = x.reshape(1, -1)
        lighter_clf_result = lighter_clf.predict(x)
        colder_clf_result = cooler_clf.predict(x)
        heater_clf_result = heater_clf.predict(x)
        rotater_clf_result = rotater_clf.predict(x)
        load_clf_result = load_clf.predict(x)
        if lighter_clf_result[0] == '1':
            y.append(light_tree.predict(x))
            j.append([
                'Compact Fluorescent Lamp', 'Incandescent Light Bulb', 'Laptop'
            ])
        if colder_clf_result[0] == '1':
            y.append(cold_tree.predict(x))
            j.append(['Fridge', 'Air Conditioner'])
        if lighter_clf_result[0] == '0' and load_clf_result[0] == 1:
            y.append(dislight_R_tree.predict(x))
            j.append(['Hairdryer', 'Heater', 'Coffee maker', 'Water kettle'])
        if heater_clf_result[0] == '1' and load_clf_result[0] == 0:
            y.append(heat_I_tree.predict(x))
            j.append(['Microwave', 'Hair Iron', 'Soldering Iron'])
        if rotater_clf_result[0] == '1' and load_clf_result[0] == 0:
            y.append(rotate_I_tree.predict(x))
            j.append([
                'Air Conditioner', 'Vacuum', 'Fan', 'Washing Machine',
                'Blender'
            ])
        if y == []:
            outputs.append('')
            tree_outputs.append([''])
            continue
        outputs.append(max(y, key=y.count))
        tree_outputs.append(sum(j, []))
    return outputs, tree_outputs


y_predict, tree_predict = classify_model(x_test)
count0 = 0
count1 = 0

for i in range(len(y_predict)):
    print(tree_predict[i])
    if y_test[i] in tree_predict[i]:
        count0 += 1
    if y_predict[i] == y_test[i]:
        count1 += 1
print(count0 / len(y_predict))
print(count1 / len(y_predict))

start_time = time.time()
print('start reading data')

x_lighter_train, y_lighter_train = read_processed_data(
    'type',
    selected_label=[
        'Compact Fluorescent Lamp', 'Incandescent Light Bulb', 'Laptop'
    ],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/training')
x_lighter_train = x_lighter_train[:, 1:]
x_lighter_validation, y_lighter_validation = read_processed_data(
    'type',
    selected_label=[
        'Compact Fluorescent Lamp', 'Incandescent Light Bulb', 'Laptop'
    ],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')
x_lighter_validation = x_lighter_validation[:, 1:]
x_lighter_trainval=np.concatenate((x_lighter_train,x_lighter_validation),axis=0)
y_lighter_trainval=np.concatenate((y_lighter_train,y_lighter_validation),axis=0) 
x_lighter_test, y_lighter_test=read_processed_data(
    'type',
    selected_label=[
        'Compact Fluorescent Lamp', 'Incandescent Light Bulb', 'Laptop'
    ],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')

x_colder_train, y_colder_train = read_processed_data(
    'type',
    selected_label=['Fridge', 'Air Conditioner'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/training')
x_colder_train = x_colder_train[:, 1:]
x_colder_validation, y_colder_validation = read_processed_data(
    'type',
    selected_label=['Fridge', 'Air Conditioner'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')
x_colder_validation = x_colder_validation[:, 1:]
x_colder_trainval=np.concatenate((x_colder_train,x_colder_validation),axis=0)
y_colder_trainval=np.concatenate((y_colder_train,y_colder_validation),axis=0) 
x_colder_test, y_colder_test = read_processed_data(
    'type',
    selected_label=['Fridge', 'Air Conditioner'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/testing')

x_dislight_R_train, y_dislight_R_train = read_processed_data(
    'type',
    selected_label=['Hairdryer', 'Heater', 'Coffee maker', 'Water kettle'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/training')
x_dislight_R_train = x_dislight_R_train[:, 1:]
x_dislight_R_validation, y_dislight_R_validation = read_processed_data(
    'type',
    selected_label=['Hairdryer', 'Heater', 'Coffee maker', 'Water kettle'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')
x_dislight_R_validation = x_dislight_R_validation[:, 1:]
x_dislight_R_trainval=np.concatenate((x_dislight_R_train,x_dislight_R_validation),axis=0)
y_dislight_R_trainval=np.concatenate((y_dislight_R_train,y_dislight_R_validation),axis=0) 

x_heat_I_train, y_heat_I_train = read_processed_data(
    'type',
    selected_label=['Microwave', 'Hair Iron', 'Soldering Iron'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/training')
x_heat_I_train = x_heat_I_train[:, 1:]
x_heat_I_validation, y_heat_I_validation = read_processed_data(
    'type',
    selected_label=['Microwave', 'Hair Iron', 'Soldering Iron'],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')
x_heat_I_validation = x_heat_I_validation[:, 1:]
x_heat_I_trainval=np.concatenate((x_heat_I_train,x_heat_I_validation),axis=0)
y_heat_I_trainval=np.concatenate((y_heat_I_train,y_heat_I_validation),axis=0) 

x_rotate_I_train, y_rotate_I_train = read_processed_data(
    'type',
    selected_label=[
        'Air Conditioner', 'Vacuum', 'Fan', 'Washing Machine', 'Blender'
    ],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/training')
x_rotate_I_train = x_rotate_I_train[:, 1:]
x_rotate_I_validation, y_rotate_I_validation = read_processed_data(
    'type',
    selected_label=[
        'Air Conditioner', 'Vacuum', 'Fan', 'Washing Machine', 'Blender'
    ],
    direaction=1,
    offset=10,
    each_lenth=10,
    source='submetered_process/validation')
x_rotate_I_validation = x_rotate_I_validation[:, 1:]
x_rotate_I_trainval=np.concatenate((x_rotate_I_train,x_rotate_I_validation),axis=0)
y_rotate_I_trainval=np.concatenate((y_rotate_I_train,y_rotate_I_validation),axis=0) 

print('finished loading data, cost %2.2fs' % (time.time() - start_time))


import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import os.path as osp
import numpy as np
from sklearn import metrics

training_file_dir = osp.join(osp.abspath(''), 'graph', 'PLAIDG', 'PLAIDG',
                             'raw')
testing_file_dir = osp.join(osp.abspath(''), 'graph', 'PLAIDG_test',
                            'PLAIDG_test', 'raw')
x_train = []
y_train = []
x_test = []
y_test = []
with open(osp.join(training_file_dir, 'table_data.txt')) as table_data:
    table_data = table_data.read().split('\n')[:-1]
    for data in table_data:
        data = data.split(',')[:-1]
        data = list(map(float, data))
        x_train.append(data[:-1])
        y_train.append(data[-1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
with open(osp.join(testing_file_dir, 'table_data.txt')) as table_data:
    table_data = table_data.read().split('\n')[:-1]
    for data in table_data:
        data = data.split(',')[:-1]
        data = list(map(float, data))
        x_test.append(data[:-1])
        y_test.append(data[-1])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

for random_state in range(20):
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_train)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_train, y_pred))
    y_test_pred = rf.predict(x_test)
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, y_test_pred))

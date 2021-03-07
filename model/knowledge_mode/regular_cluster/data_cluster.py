from sklearn.cluster import KMeans
from sklearn import tree
import numpy as np
import time
import sys
import pydotplus
from IPython.display import Image, display
import matplotlib.pyplot as plt
sys.path.append(r'data/')
from read_PLAID_data import read_processed_data, get_feature_name

process_start_time = time.time()
label_transformer = {'I': 0, 'R': 1, 'NL': 0}
x, y = read_processed_data('load',
                           direaction=1,
                           offset=30,
                           Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x = x[:, 1:]
x_cluster = np.zeros([x.shape[0], x.shape[1]])
for i in range(x.shape[1]):
    start_cluster_time = time.time()
    km = KMeans(n_clusters=3)
    xi = x[:, i]
    xi = xi.reshape(-1, 1)
    km.fit(xi)
    x_cluster[:, i] = km.labels_
    cost_time = time.time() - start_cluster_time
    print('[%02d/%02d]:%2.2f s' % (i, x.shape[1], cost_time))
np.savetxt('model/knowledge_mode/x_cluster.csv', x_cluster, delimiter=',')
np.savetxt('model/knowledge_mode/y_label.csv', y, delimiter=',')
np.savetxt('model/knowledge_mode/x.csv', x, delimiter=',')

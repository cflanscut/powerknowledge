import numpy as np
import time
import sys
sys.path.append(r'data/')
from read_PLAID_data import read_processed_data

process_start_time = time.time()
label_transformer = {'I': 0, 'R': 1, 'NL': 0}
x, y = read_processed_data('load',
                           direaction=1,
                           offset=30,
                           Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x = x[:, 1:]
np.savetxt('model/knowledge_model/loadtype/y_label.csv', y, delimiter=',')
np.savetxt('model/knowledge_model/loadtype/x.csv', x, delimiter=',')

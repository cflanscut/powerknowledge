import numpy as np
import time
import sys
sys.path.append(r'data/')
from read_PLAID_data import read_processed_data

process_start_time = time.time()
label_transformer = {'0': 0, '1': 1}
x, y = read_processed_data('is_light',
                           type_header='extra label',
                           direaction=1,
                           offset=30,
                           Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x = x[:, 1:]
np.savetxt('model/knowledge_model/lighter/y_label.csv', y, delimiter=',')
np.savetxt('model/knowledge_model/lighter/x.csv', x, delimiter=',')

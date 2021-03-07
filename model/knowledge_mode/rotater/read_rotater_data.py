import numpy as np
import time
import sys
sys.path.append(r'dataset/')
from classifieddata import read_processed_data

process_start_time = time.time()
label_transformer = {'0': 0, '1': 1}
x, y = read_processed_data('is_rotate',
                           type_header='extra label',
                           direaction=1,
                           offset=30,
                           Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x = x[:, 1:]
np.savetxt('model/knowledge_mode/rotater/y_label.csv', y, delimiter=',')
np.savetxt('model/knowledge_mode/rotater/x.csv', x, delimiter=',')

import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import pandas as pd
sys.path.append(r'dataset/')
sys.path.append(r'model/')
from classifieddata import read_processed_data
from linear import SingleLayerModel, DoubleLayerModel, MultiLayerModer

process_start_time = time.time()
label_transformer = {'I': 0, 'R': 1, 'NL': 2}
x, y = read_processed_data('load', Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if x_std[j] != 0:
            x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]
# for i, label in enumerate(y):
#     if label == 'I':
#         y[i] = '0'
#     elif label == 'R':
#         y[i] = '1'
#     elif label == 'NL':
#         y[i] = '2'

x_train = torch.tensor(x).float()
y_train = torch.tensor(y.astype(float)).float()

dataset = Data.TensorDataset(x_train, y_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

# def get_acc(outputs, labels):
#     counts = 0
#     for index, output in enumerate(outputs):
#         if torch.argmax(output) == labels[index]:
#             counts += 1
#     return counts / len(labels)

# model = SingleLayerModel(x_train.size()[1], 3)
model = DoubleLayerModel(x_train.size()[1], 30, 3)
# model = MultiLayerModer(x_train.size()[1], 3)
optim = torch.optim.SGD(model.parameters(), lr=0.05)
loss_f = nn.CrossEntropyLoss()
epoch_num = 200
for epoch in range(epoch_num):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_acc = 0
    for i, (x_epoch, y_epoch) in enumerate(dataloader):
        optim.zero_grad()
        outputs = model(x_epoch)
        loss = loss_f(outputs, y_epoch.type(torch.LongTensor))

        loss.backward()
        optim.step()

        epoch_acc += np.sum(
            np.argmax(outputs.cpu().data.numpy(), axis=1) == y_epoch.numpy())
        epoch_loss += loss.item()

    print('[%03d/%03d] %2.2f s TA: %3.6f Ls: %3.6f' %
          (epoch + 1, epoch_num, time.time() - epoch_start_time,
           epoch_acc / y.__len__(), epoch_loss / y.__len__()))

test_predict = model(x_train)
test_label = np.argmax(test_predict.data.numpy(), axis=1)
test_result = pd.concat([
    pd.DataFrame({'predict': test_label}),
    pd.DataFrame({'label': y_train.numpy()})
],
                        axis=1)
test_result.to_csv('model/knowledge_mode/load_test_result.csv', index=True, sep=',')

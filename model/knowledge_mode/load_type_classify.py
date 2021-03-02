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


def confusion_matrix(preds, labels, conf_matrix):
    for p, l in zip(preds, labels):
        conf_matrix[p, l] += 1
    return conf_matrix


process_start_time = time.time()
label_transformer = {'I': 0, 'R': 1, 'NL': 0}
feature_select = ['P_F', 'z_hp1']
x, y = read_processed_data('load',
                           feature_select=feature_select,
                           direaction=1,
                           offset=30,
                           Transformer=label_transformer)
print('finished reading data, cost %2.2f s' %
      (time.time() - process_start_time))
x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if x_std[j] != 0:
            x[i][j] = (x[i][j] - x_mean[j]) / x_std[j]

x_train = torch.tensor(x).float()
y_train = torch.tensor(y.astype(float)).float()

dataset = Data.TensorDataset(x_train, y_train)
dataloader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

model = SingleLayerModel(x_train.size()[1], 2)
# model = DoubleLayerModel(x_train.size()[1], 3, 2)
# model = MultiLayerModer(x_train.size()[1], 2)
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
test_predict = np.argmax(test_predict.data.numpy(), axis=1)
test_result = pd.concat([
    pd.DataFrame({'predict': test_predict}),
    pd.DataFrame({'label': y_train.numpy()})
],
                        axis=1)
conf_matrix = np.zeros([2, 2])
conf_matrix = confusion_matrix(test_predict,
                               y_train.numpy().astype(int),
                               conf_matrix=conf_matrix)
conf_matrix = conf_matrix.astype(int)
test_result = pd.concat([
    test_result,
    pd.DataFrame({
        'label': ['I/NL', 'R'],
        'I/NL': conf_matrix[0, :],
        'R': conf_matrix[1, :],
        # 'NL': conf_matrix[2, :]
    })
],
                        axis=1)
test_result.to_csv('model/knowledge_mode/load_test_result.csv',
                   index=True,
                   sep=',')

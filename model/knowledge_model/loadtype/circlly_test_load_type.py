import torch.utils.data as Data
import torch
import torch.nn as nn
import numpy as np
import time
import sys
sys.path.append(r'model/')
sys.path.append(r'data/')
from read_PLAID_data import read_processed_data
from linear import SingleLayerModel, DoubleLayerModel, MultiLayerModer

torch.cuda.set_device(1)


def train_f(x, y):
    train_start_time = time.time()
    x_train = torch.tensor(x).float()
    y_train = torch.tensor(y.astype(float)).float()

    dataset = Data.TensorDataset(x_train, y_train)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=128, shuffle=True)

    model = SingleLayerModel(x_train.size()[1], 2)
    # model = DoubleLayerModel(x_train.size()[1], 3, 2)
    # model = MultiLayerModer(x_train.size()[1], 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_f = nn.CrossEntropyLoss()
    epoch_num = 50
    for epoch in range(epoch_num):
        epoch_loss = 0
        epoch_acc = 0
        for i, (x_epoch, y_epoch) in enumerate(dataloader):
            optim.zero_grad()
            outputs = model(x_epoch)
            loss = loss_f(outputs, y_epoch.type(torch.LongTensor))

            loss.backward()
            optim.step()

            epoch_acc += np.sum(
                np.argmax(outputs.cpu().data.numpy(), axis=1) ==
                y_epoch.numpy())
            epoch_loss += loss.item()
    train_time = time.time() - train_start_time
    return epoch_acc, train_time


def confusion_matrix(preds, labels, conf_matrix):
    for p, l in zip(preds, labels):
        conf_matrix[p, l] += 1
    return conf_matrix


process_start_time = time.time()
label_transformer = {'I': 0, 'R': 1, 'NL': 0}
x, y = read_processed_data('load',
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

feature_len = x.shape[1]
data_len = x.shape[0]

acc_matrix = np.zeros([feature_len, feature_len])
for a in range(feature_len):
    x_select = np.zeros([data_len, 2])
    x_select[:, 0] = x[:, a]
    for b in range(a, feature_len):
        x_select[:, 1] = x[:, b]
        acc, t_time = train_f(x_select, y)
        print('feature index %03d with %03d: acc %2.6f, cost %2.2f s ' %
              (a, b, acc / data_len, t_time))
        acc_matrix[a, b] = acc / data_len
np.savetxt('model/knowledge_mode/acc_matrix.csv', acc_matrix, delimiter=',')

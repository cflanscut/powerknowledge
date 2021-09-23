import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 数据集加载
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')


# 网络定义
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16) # 节点特征为num_node_feature,经过计算后变成16
        self.conv2 = GCNConv(16, dataset.num_classes)   # 节点特征为16，经过计算变成类别数（再softmax）

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)  # 计算不影响图的结构，只影响节点的值
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)  # 声明模型，并输入至gpu
data = dataset[0].to(device) # 读取一张图数据，输入至gpu
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4) # 声明优化器

# 网络训练
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 测试
model.eval()
_, pred = model(data).max(dim=1)
correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

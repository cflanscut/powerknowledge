
from graph.PLAID_dataset import PLAIDDataset

datasets = PLAIDDataset(root='~/graph/PLAIDG',
                        name='PLAIDG',
                        use_node_attr=True,
                        use_edge_attr=True)

data = datasets[1]
print(data)
print(data.edge_index)
print(data.x)
print(data.y)

---
layout: post
title:  "Pytorch Geometric"
date:   2023-04-29 11:00:20 +0700
categories: [RecSys]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['<span>$$','<span>$$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Pytorch Geometric
- Link: https://pytorch-geometric.readthedocs.io/en/latest/

현재 Post는 PyG(Pytorch Geometric)에 관련하여 알아보는 Post 입니다. GNN에서 많은 기능을 지원하고, 다른 논문들에서 기본적으로 많이 사용하므로 앞으로 논문 구현 및 공부하기 위하여 해당 패키지에 대해 알아봅니다.

### PyG Tutorials

먼저, PyG의 Tutorial을 통하여, PyG의 특징과 실제 예제롤 확인하여 보자.

**Import Library**


```python
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import scatter

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset, ShapeNet
```

### Data Handling of Graphs

Graph는 pairwise relation (edges) between objects(nodes)로 이루워져 있다. 해당 Dataset에 대하여 PyG는 <code>torch_geometric.data.Data</code>로서 제공한다.

**<code>torch_geometric.data.Data</code> Argument**. 

- <code>data.x</code>: Node feature matrix with shape <code>[num_nodes, num_node_features]</code>
- <code>data.edge_index</code>: Graph connectivity with shape [2, num_edges] and type <code>torch.long</code>
- <code>data.edge_attr</code>: Edge feature matrix with shape <code>[num_edges, num_edge_features]</code>
- <code>data.y</code>: Target to train against with shape <code>[num_nodes, *]</code>
- <code>data.pos</code>: Node position matrix with shape <code>[num_nodes, num_dimensions]</code>

![png](https://pytorch-geometric.readthedocs.io/en/latest/_images/graph.svg)

위와 같은 Figure는 아래와 같은 Dataset으로서 표현 가능하다.


```python
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
data
```




    Data(x=[3, 1], edge_index=[2, 4])



해당 사항에서 주요한 점은 <code>edge_index</code>를 <code>[num_edges, 2]</code>의 Shape로서 정의하게 되면, 단순히 transpose 뿐만 아니라, contiguous또한 선언해야 한다.

- 참조: <a href="https://jimmy-ai.tistory.com/122">Torch의 Contiguous에 대해서</a>


```python
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())
data
```




    Data(x=[3, 1], edge_index=[2, 4])



해당 dataset을 사용할 수 있는 Format인지 확인하기 위해서는 아래와 같이 <code>.validate(raise_on_error=True)</code>로서 확인 가능하다.


```python
data.validate(raise_on_error=True)
```




    True



### Common Benchmark Datasets

PyG에서는 여러가지 Benchmark Datasets를 제공한다. 그 중 많이 사용하는 Cora Dataset에 대하여 알아보자.


```python
## Dataset ##
print('-'*20+'Dataset'+'-'*20)
dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
print('Number of Nodes: {}'.format(dataset.x.shape[0]))
print('Number of Edges: {}'.format(dataset.edge_index.shape[0]))
print('Number of Classes: {}'.format(dataset.num_classes))
print('Number of Node Features: {}'.format(dataset.num_node_features))

## Data ##
print('\n\n\n'+'-'*20+'Data'+'-'*20)
data = dataset[0]
print('Is Undirected?: {}'.format(data.is_undirected()))
print(data)
print('Number of Train Mask // Shape: {}, Sum: {}'.format(
len(data.train_mask), data.train_mask.sum()))
print('Number of Validation Mask // Shape: {}, Sum: {}'.format(
len(data.val_mask), data.val_mask.sum()))
print('Number of Test Mask // Shape: {}, Sum: {}'.format(
len(data.test_mask), data.test_mask.sum()))
```

    --------------------Dataset--------------------
    Cora()
    Number of Nodes: 2708
    Number of Edges: 2
    Number of Classes: 7
    Number of Node Features: 1433
    
    
    
    --------------------Data--------------------
    Is Undirected?: True
    Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    Number of Train Mask // Shape: 2708, Sum: 140
    Number of Validation Mask // Shape: 2708, Sum: 500
    Number of Test Mask // Shape: 2708, Sum: 1000


기본적인 특성을 말고 Index를 하거나, Permutation을 하는 방법은 아래와 같다.


```python
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
data = dataset[0]

print('Train Dataset')
train_dataset = dataset[:540]
print(train_dataset)
print('Value')
print(train_dataset.x[100:103, :])

print('\n\nTest Dataset')
test_dataset = dataset[540:]
print(test_dataset)
print('Value')
print(test_dataset.x[100:103, :])

print('\n\nAfter Shuffle Train Dataset')
train_dataset = train_dataset.shuffle()
print(train_dataset)
print('Value')
print(train_dataset.x[100:103, :])

print('\n\nPermutation')
perm = torch.randperm(len(train_dataset))
train_dataset = train_dataset[perm]
print(train_dataset)
print('Value')
print(train_dataset.x[100:103, :])
```

    Train Dataset
    ENZYMES(540)
    Value
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])
    
    
    Test Dataset
    ENZYMES(60)
    Value
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])
    
    
    After Shuffle Train Dataset
    ENZYMES(540)
    Value
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])
    
    
    Permutation
    ENZYMES(540)
    Value
    tensor([[1., 0., 0.],
            [1., 0., 0.],
            [1., 0., 0.]])


### Mini-baches

Mini-batch란 PyTorch에서 사용하는 것 처럼 사용하기 위한 방법이다. PyTorch와 동일하게 병렬로 처리 가능하여, 빠른 수행이 가능하다. 

<a href="https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html">공식 홈페이지</a>에서 사용하는 Dataset과 DataLoader의 사용 방법은 아래와 같다.

```python
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)
```

실제 Dataset을 Custom하여 확인하자.

**현재 구축한 Dataset은 아래와 같은 구성으로서 이루워져 있다.**
- <code>dataset_list</code>: 100개의 Graph를 포함하고 있는 List. 각각의 원소는 한개의 Graph를 의미한다.
- <code>x</code>: 한개의 Graph당 100개의 Node를 가지고 있다. 각각의 Node는 5개의 Feature를 가지고 있다.
- <code>edge</code>: 한개의 Graph당 100개의 Edge를 가지고 있다.


```python
dataset_list = []

for i in range(100):
    edge_origin = torch.randint(100, (100, 1))
    edge_destination = torch.randint(100, (100, 1))
    edge_index = torch.stack([edge_origin.squeeze(), edge_destination.squeeze()], 0).type(torch.LongTensor)

    x = torch.rand((100, 5), dtype=torch.float)
    dataset = Data(x=x, edge_index=edge_index)
    dataset_list.append(dataset)
```

아래 결과를 살펴보게 되면, PyG의 DataLoader는 다음과 같은 특징을 가지고 있다.

1. DataLoader에 담겨야 하는 내용은 Graph를 List형태로 담아서 선언해야 한다.
2. Batch로서 가져오게 되면, DataLoader안에서 Batch개 만큼의 Graph를 가져온다.
3. 각 Graph는 Concat하여 반환해 준다. 즉, 100(Node)x5(Feature)의 Graph를 Batchsize=20으로서 불러오게 되면 -> 2000 x 5가 된다.
4. <code>batch.batch</code>는 각 Node가 속한 Graph를 의미하게 된다.


```python
loader = DataLoader(dataset_list, batch_size=20, shuffle=False)

for batch in loader:
    print('Batch Index: ', batch.batch)
    print('Number of Graphs: ', batch.num_graphs)
    print('Data Feature: ', batch.x.shape)
```

    Batch Index:  tensor([ 0,  0,  0,  ..., 19, 19, 19])
    Number of Graphs:  20
    Data Feature:  torch.Size([2000, 5])
    Batch Index:  tensor([ 0,  0,  0,  ..., 19, 19, 19])
    Number of Graphs:  20
    Data Feature:  torch.Size([2000, 5])
    Batch Index:  tensor([ 0,  0,  0,  ..., 19, 19, 19])
    Number of Graphs:  20
    Data Feature:  torch.Size([2000, 5])
    Batch Index:  tensor([ 0,  0,  0,  ..., 19, 19, 19])
    Number of Graphs:  20
    Data Feature:  torch.Size([2000, 5])
    Batch Index:  tensor([ 0,  0,  0,  ..., 19, 19, 19])
    Number of Graphs:  20
    Data Feature:  torch.Size([2000, 5])


위와 같은 이유로 아래와 같이 평균을 취하여 사용 가능하다.


```python
# Average
x = scatter(batch.x, batch.batch, dim=0, reduce='mean')
print(x.size())
```

    torch.Size([20, 5])


### Data Transform

Data Transform은 종류가 매우 많다. 해당 종류에 대해서는 <a href="https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html">PyG 공식 홈페이지</a>를 참조 하자.

먼저, 관심있는 Dataset Split이다. Graph기반의 Model은 크게 Node Classification, Link Prediction이 존재하게 된다. 먼저 Dataset은 아래와 같이 정의된다.


```python
from torch_geometric.datasets import KarateClub
from torch_geometric.transforms import RandomLinkSplit

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)
print('Shape of Dataset: {}'.format(dataset.x.shape))
print('Number of Edges: {}'.format(dataset.edge_index.shape))
```

    Cora()
    Shape of Dataset: torch.Size([2708, 1433])
    Number of Edges: torch.Size([2, 10556])


### RandomLinkSplit

Edge를 대상으로 Train, Validation, Test set을 Split 진행 합니다.

**Appendix: <code>RandomLinkSplit</code>**: https://github.com/pyg-team/pytorch_geometric/issues/3668

해당 되는 RandomLinkSplit에 대하여 토론을 나눈 주소 입니다. Link Prediction의 개념을 잡는데 매우 주요한 부분인 것 같습니다.

**Edge Index**


```python
transform = T.Compose([
    T.NormalizeFeatures(),
    # T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False),
])

train_dataset, val_dataset, test_dataset = transform(dataset[0])

print('Number of Train Edges: {}'.format(train_dataset.edge_index.shape))
print('Number of Validation Edges: {}'.format(val_dataset.edge_index.shape))
print('Number of Test Edges: {}'.format(test_dataset.edge_index.shape))
```

    Number of Train Edges: torch.Size([2, 8448])
    Number of Validation Edges: torch.Size([2, 8448])
    Number of Test Edges: torch.Size([2, 9502])


**Edge label Index**


```python
print('Number of Train Edges: {}'.format(train_dataset.edge_label_index.shape))
print('Number of Validation Edges: {}'.format(val_dataset.edge_label_index.shape))
print('Number of Test Edges: {}'.format(test_dataset.edge_label_index.shape))
```

    Number of Train Edges: torch.Size([2, 4224])
    Number of Validation Edges: torch.Size([2, 1054])
    Number of Test Edges: torch.Size([2, 1054])


**Edge label**


```python
print('Number of Train Edges: {}'.format(train_dataset.edge_label.shape))
print('Number of Validation Edges: {}'.format(val_dataset.edge_label.shape))
print('Number of Test Edges: {}'.format(test_dataset.edge_label.shape))
```

    Number of Train Edges: torch.Size([4224])
    Number of Validation Edges: torch.Size([1054])
    Number of Test Edges: torch.Size([1054])


위의 결과를 살펴보게 되면, edge label기준으로는 모두 잘 split한 것을 알 수 있다. 하지만, edge_index는 조금 다른 숫자를 가지고 있는것을 알 수 있다. 

이러한 이유는 Graph는 주변 정보를 가져오게 되는 Message Passing개념이 있기 때문이다.

즉, Link를 단순히 Split하는 것 뿐만 아니라 Node까지 끝어야 정보 전달이 되지 않기 때문이다.

실제 결과를 살펴보면, 아래와 같이 Validation, Test에만 존재하는 Node들이 존재하며, 서로 겹치지 않게 고려하여여 Split 해준다.


```python
train_node = train_dataset.edge_label_index.unique()
val_node = val_dataset.edge_label_index.unique()
test_node = test_dataset.edge_label_index.unique()

only_train_node = list(set(train_node.numpy()) - set(val_node.numpy()) - set(test_node.numpy()))
only_val_node = list(set(val_node.numpy()) - set(train_node.numpy()) - set(test_node.numpy()))
only_test_node = list(set(test_node.numpy()) - set(train_node.numpy()) - set(val_node.numpy()))

print('Number of only train node: {}'.format(len(only_train_node)))
print('Number of only validation node: {}'.format(len(only_val_node)))
print('Number of only test node: {}'.format(len(only_test_node)))
```

    Number of only train node: 632
    Number of only validation node: 39
    Number of only test node: 33


### RandomNodeSplit

**RandomNodeSplit의 결과를 살펴보게 되면, Train, Validation, Test set으로 완전한 Split이 아닌 mask를 Dataset안에 담는 형식으로 Return 한다.**

즉, Edge정보는 모두 동일하게 사용하되, Node만 Split하는 형태이다.


```python
transform = T.Compose([
    T.NormalizeFeatures(),
    # T.ToDevice(device),
    T.RandomNodeSplit(num_val=0.1, num_test=0.1),
])

node_dataset = transform(dataset[0])

print('Number of Train Node: {}'.format(sum(node_dataset.train_mask)))
print('Number of Validation Node: {}'.format(sum(node_dataset.val_mask)))
print('Number of Test Node: {}'.format(sum(node_dataset.test_mask)))
```

    Number of Train Node: 2166
    Number of Validation Node: 271
    Number of Test Node: 271


### Learning Methods on Graphs

간단한 GCN Layer로서 Graph를 학습하는 과정이다. (Node Classification Task)

기존 Troch와 다른 점은 <code>dataset[data.train_mask]</code>같은 형태로서 Train Dataset을 구축한다는 것 이다.


```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load Dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```

**GCN Layer**


```python
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

**Model Train**


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

**Validation**


```python
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
```

    Accuracy: 0.8020


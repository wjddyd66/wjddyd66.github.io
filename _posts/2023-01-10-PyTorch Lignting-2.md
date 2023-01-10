---
layout: post
title:  "PyTorch Lightning Ch2-LightningDataModule"
date:   2023-01-10 08:00:20 +0700
categories: [PytorchLightning]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## PyTorch Lightning Module 

개인적으로 생각하였을 때, PyTorch Lightning을 사용하기 위한 주요한 Module은 3가지라고 생각된다.

참조: 앞으로의 예제 Code는 <a href="https://wjddyd66.github.io/paper/MConfident-net(28)/">Multimodal Dynamics: Dynamical Fusion for Trustworthy Multimodal Classification</a>로서 작성하였습니다.

### LightningDataModule API
- Link: https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html

To define a DataModule the following methods are used to create train/val/test/predict dataloaders:

- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#prepare-data">prepare_data</a> (how to download, tokenize, etc…)
- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#setup">setup</a> (how to split, define dataset, etc…)
- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#train-dataloader">train_dataloader</a>
- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#val-dataloader">val_dataloader</a>
- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#test-dataloader">test_dataloader</a>
- <a href="https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#predict-dataloader">predict_dataloader</a>

위와 같이 LightningDataModule이 정의 된다. 기본적으로 우리가 DeepLearning에 사용하기 위한 Data를 준비한다고 하면, 다음과 같은 과정을 거치게 된다.
1. Dataset을 다운로드 받는다. (prepare_data)
2. Dataset을 Preprocessing 및 Data를 Split한다. (setup)
3. Training Dataset을 준비한다. (train_dataloader)
4. Validation Dataset을 준비한다. (val_dataloader)
5. Test Dataset을 준비한다. (test_dataloader)
6. Inference용 Dataset을 준비한다. (Additional). (predict_dataloader)

실제 Code에 적용하기 위하여 LightningDataModule은 아래와 같이 정의하였다.

**Import Library**


```python
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
```

**Define Dataset by using PyTorch**


```python
# Dataset
class TCP_Dataset(Dataset):
    def __init__(self, data_path, status):
        # Load Label
        self.labels = np.load(os.path.join(data_path, "Label_" + status + ".npy"))
        self.labels = self.labels.astype(int)

        # Load Data
        num_view = 3
        self.data_list = []
        for i in range(1, num_view + 1):
            self.data_list.append(
                pd.read_csv(os.path.join(data_path, 'M' + str(i) + '_' + status + '.csv'), index_col=0).values)
        self.data_list = np.array(self.data_list)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data_list = self.data_list[:, idx, :]

        return data_list, label
```

**Define Dataset by using PyTorch Lightning**


```python
# Torchlighting DataLoader
class TCP_DataModule(LightningDataModule):
    def __init__(self, data_type, cv, batch_size, num_workers):
        super().__init__()
        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cv = cv

    def setup(self, stage: str):
        data_path = os.path.join('/data/jyhwang/TCP/Data/Preprocessing/', self.data_type, 'cv' + str(self.cv + 1))

        # Define Dataset
        if stage == 'fit':
            self.train_dataset = TCP_Dataset(data_path, 'train')
            self.val_dataset = TCP_Dataset(data_path, 'validation')

        if stage == 'test' or stage is None:
            self.test_dataset = TCP_Dataset(data_path, 'test')

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
```

**Check Dataset**


```python
print('ROSMAP')
dataloader = TCP_DataModule(data_type='ROSMAP', cv=0, batch_size=128, num_workers=8)

dataloader.setup('fit')
# use data
print('\n\nTrain Dataset')
for data, label in dataloader.train_dataloader():
    print('Data Shape: {}, Min: {:.1f}, Max: {:.1f}'.format(data.shape, data.max(), data.min()))
    print('Category: {}'.format(set(label.detach().numpy())))
    
# use data
print('\n\nValidation Dataset')
for data, label in dataloader.val_dataloader():
    print('Data Shape: {}, Min: {:.1f}, Max: {:.1f}'.format(data.shape, data.max(), data.min()))
    print('Category: {}'.format(set(label.detach().numpy())))

dataloader.setup('test')
# use data
print('\n\nTest Dataset')
for data, label in dataloader.test_dataloader():
    print('Data Shape: {}, Min: {:.1f}, Max: {:.1f}'.format(data.shape, data.max(), data.min()))
    print('Category: {}'.format(set(label.detach().numpy())))
```

    ROSMAP
    
    
    Train Dataset
    Data Shape: torch.Size([128, 3, 300]), Min: 1.0, Max: 0.0
    Category: {0, 1}
    Data Shape: torch.Size([128, 3, 300]), Min: 1.0, Max: 0.0
    Category: {0, 1}
    Data Shape: torch.Size([44, 3, 300]), Min: 1.0, Max: 0.0
    Category: {0, 1}
    
    
    Validation Dataset
    Data Shape: torch.Size([38, 3, 300]), Min: 1.0, Max: 0.0
    Category: {0, 1}
    
    
    Test Dataset
    Data Shape: torch.Size([38, 3, 300]), Min: 1.0, Max: 0.0
    Category: {0, 1}


**위의 Code에서 주의하여야 할 점은 <code>setup(self, stage: str)</code>이다.**  
해당 함수 안에서 stage라는 string인자는 무조건 받게 되어있으며, training, validation dataset은 stage가 fit이며, test dataset은 stage가 test이다. 해당 방법으로 무조건 정의하는 이유는 PyTorch Lightning의 Trainer라는 Module과 같이 사용하기 위해서 이다.  

**Appendix1. Inference Dataset**  
위의 Code에서는 적지 않았으나, Inference Dataset을 사용하기 위해서는 아래의 예시와 같이 <code>predict_dataloader(self)</code>가 필요하고, <code>setup(self, stage:str)</code>안에서는 <code>stage=='predict'</code>로서 받게 된다.

**Torch Lightning Example Code**


```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
```

**Appendix2. Type hinting**  
출처: <a href="https://www.daleseo.com/python-type-annotations/">daleseo Blog</a>

동적(dynamic) 프로그래밍 언어인 파이썬에서는 인터프리터(interpreter)가 코드를 실행하면서 타입(type)을 추론하여 체크합니다. 또한 **파이썬에서 변수의 타입(type)은 고정되어 있지 않기 때문에 개발자가 원하면 자유롭게 바꿀 수 있습니다.**


```python
no = 1
print(type(no))
no = "1"
print(type(no))
```

    <class 'int'>
    <class 'str'>


위 코드를 보면 no 변수의 타입이 처음에는 int였다가 str으로 바뀐 것을 알 수 있습니다. 자바와 같은 정적(static) 프로그래밍 언어에서는 상상도 하기 힘든 일이며 이런 코드는 컴파일(compile)조차 되지 않습니다.

따라서, 여러 언어를 함께 사용하는 대형 프로젝트에서는 Interpreter기반의 python이 문제를 발생시킬 수 있다. **이러한 문제점을 해결할 수 있는 것이 Type hinting이다.**

**타입 힌팅에서는 타입 어노테이션(annotation)이라는 새로운 방법으로 파이썬 코드의 타입 표시를 표준화합니다. 따라서 코드 편집기(IDE)나 린터(linter)에서도 해석할 수 있도록 고안되었으며, 코드 자동 완성이나 정적 타입 체킹에도 활용되고 있습니다.**

이러한 타입 어노테이션은 크게 2가지에서 사용된다.

**1. 변수 타입 어노테이션**  
먼저 매우 간단한 변수에 타입 어노테이션을 추가하는 방법에 대해서 알아보겠습니다. 변수 이름 뒤에 콜론(:)을 붙이고 타입을 명시해주면 됩니다.


```python
name: str = "John Doe"
print(type(name))
```

    <class 'str'>


**2. 함수 타입 어노테이션**  
함수에 타입 힌탕을 적용할 때는 인자 타입과 반환 타입, 이렇게 두 곳에 추가해줄 수 있습니다.

인자에 타입 어노테이션을 추가할 때는 변수와 동일한 문법을 사용하며, 반환값에 대한 타입을 추가할 때는 화살표(->)를 사용합니다.


```python
def stringify(num: int) -> str:
    return str(num)
```

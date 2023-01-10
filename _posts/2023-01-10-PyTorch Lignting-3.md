---
layout: post
title:  "PyTorch Lightning Ch3-LightningModule"
date:   2023-01-10 09:00:20 +0700
categories: [PytorchLightning]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## LightningModule API
- Link: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html

A LightningModule organizes your PyTorch code into 6 sections:

- Computations (init).
- Train Loop (training_step)
- Validation Loop (validation_step)
- Test Loop (test_step)
- Prediction Loop (predict_step)
- Optimizers and LR Schedulers (configure_optimizers)


LightningModule은 크게 2가지로 구분된다.

1. 모델의 기본적인 구조 정의 (Computations, Forward, Optimizers and LR Schedulers)
2. 모델 학습 루프 (Train Loop, Validation Loop, Test Loop, Prediction Loop)

아래 정의한 LightningModule을 보고, Code에 대한 Function을 설명해보자.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from Utils import evaluate

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

# MMDynamic - TorchLighting
class MMDynamic(LightningModule):
    def __init__(self, in_dim, num_class, binary, config):
        super().__init__()
        self.save_hyperparameters()
        self.views = len(in_dim)
        self.classes = num_class
        self.binary = binary

        # Define Hyperparameters
        hidden_dim = [config['hidden_dim']]
        self.lr = config['lr']
        self.reg = config['reg']
        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=0.5))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.test_result = None

    def forward(self, data_list, status='validation'):
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[:, view, :]))
            feature[view] = data_list[:, view, :] * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], 0.5, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)

        if status == 'train':
            return FeatureInfo, TCPLogit, TCPConfidence, MMlogit

        else:
            return MMlogit

    def training_step(self, batch, batch_idx):
        data_list, label = batch
        FeatureInfo, TCPLogit, TCPConfidence, MMlogit = self(data_list.float(), status='train')

        MMLoss = torch.mean(self.criterion(MMlogit, label))
        for view in range(self.views):
            MMLoss = MMLoss + torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + self.criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss

        return MMLoss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)

        return [opt], [scheduler]

    def validation_step(self, val_batch, batch_idx):
        data_list, label = val_batch
        MMlogit = self(data_list.float())
        MMLoss = torch.mean(self.criterion(MMlogit, label))

        return {'val_loss': MMLoss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss)

        return {'ptl/val_loss': avg_loss}

    def test_step(self, test_batch, batch_idx):
        data_list, label = test_batch
        MMlogit = self(data_list.float())
        prob = F.softmax(MMlogit, dim=1)
        return {'test_prob': prob, 'test_label': label}

    def test_epoch_end(self, outputs):

        test_prob = torch.cat([x['test_prob'] for x in outputs]).data.cpu().numpy()
        test_label = torch.cat([x['test_label'] for x in outputs]).data.cpu().numpy()
        test_metric_1, test_metric_2, test_metric_3 = evaluate(test_label, test_prob, binary=self.binary)

        self.test_result = {'test_metric_1': test_metric_1, 'test_metric_2': test_metric_2, 'test_metric_3': test_metric_3}

        return self.test_result
```

### 모델의 기본적인 구조 정의 (Computations, Forward, Optimizers and LR Schedulers)

- **1. <code>__init__</code>**: 기본적으로 Model을 정의하는 부분이다. 기존의 PyTorch와 동일하다.  
여기서 주의해야할 점은 <code>config</code>라는 dict type의 인자를 input으로 받는다. (추후에 hyperparameter search에서 사용됨.)
- **2. <code>forward(self, data_list, status='validation')</code>**: 기존의 PyTorch와 동일한 forward역할을 한다.
- **3. <code>configure_optimizers(self)</code>**: Optimizer로서 model의 parameter를 학습함과 동시에 Scheduler를 반환한다. 순서는 [Optimizer], [Scheduler]로서 최대 2개까지 Return가능하다.

### 모델 학습 루프 (Train Loop, Validation Loop, Test Loop, Prediction Loop)
모델 학습은 크게 Train, Validation, Test, Prediction으로 나누어지나, 모두 **validation_step** or **validation_epoch_end**으로서 이루워진다.

- **1. <code>validation_step(self, val_batch, batch_idx)</code>**: Input은 모두 Batch와 Index를 받으며, 매 Batch마다 실행된다.
- **2. <code>validation_epoch_end(self, outputs)</code>**: Input은 모든 validation_step의 Outputs를 받게 되며, 매 Epoch마다 실행한다.

위의 Code중에서 주의하여야 할 점은, <code>avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()</code>와 <code>self.log("ptl/val_loss", avg_loss)</code>이다.  

<code>avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()</code>은 모든 Validation의 Loss값의 평균을 구하는 과정이다.

<code>self.log("ptl/val_loss", avg_loss)</code>은 Log에 "ptl/val_loss"라는 변수의 이름에 avg_loss값을 저장한다는 의미이다. 해당 과정은 Hyperparameter Search나, 나중에 Tensorboard와 같은 Tool로서 결과 확인을 하는데 주로 사용된다. (나중에 Posting)

**Appendix. <code>self.automatic_optimization</code>**  
LightningModule은 <code>self.automatic_optimization = True</code>로서 Default Setting되어 있다. <code>self.automatic_optimization</code>은 PyTorch에서 <code>optimizer.zero_grad()</code> -> <code>loss.backward()</code> -> <code>optimizer.step()</code> 과정을 수행해주는 Option이다. 이러한 Option을 False로 하면 아래와 같이 Optimization Code를 작성하여야 한다.


```python
class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()
```

<code>self.automatic_optimization</code>을 False로 하는 경우는 아래와 같다.
1. Learning Rate를 조절하는 Scheduler를 Epoch단위가 아닌 Batch 단위로서 바꾸는 경우
2. Loss를 계산하고 추가적인 과정이 필요한 경우.

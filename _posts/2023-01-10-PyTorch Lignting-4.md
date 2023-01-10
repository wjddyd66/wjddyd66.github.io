---
layout: post
title:  "PyTorch Lightning Ch4-Trainer"
date:   2023-01-10 10:00:20 +0700
categories: [PytorchLightning]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## Trainer API
- Link: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html


- Automatically enabling/disabling grads
- Running the training, validation and test dataloaders
- Calling the Callbacks at the appropriate times
- Putting batches and computations on the correct devices

Trainer API는 Argument가 많다. 따라서 하나하나 따라해 보면서 진행을 해보자.

**Import Library**


```python
from Utils import TCP_DataModule
from MLDRL_Lighting import MMDynamic

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
```

**DataLoad**


```python
dataloader = TCP_DataModule(data_type='ROSMAP', cv=0, batch_size=32, num_workers=10)
```

**Define Model**


```python
# Model Define
config = {
    "lr": 1e-5,
    "reg": 1e-5,
    "hidden_dim": 50}

model = MMDynamic([300, 300, 300], 2, True, config)
```

**Define Callback**


```python
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
```

**Model Train & Validation**


```python
trainer = Trainer(
    logger=TensorBoardLogger(save_dir="./Logs", name="exp"), # 실험 로거 정의
    accelerator='gpu',
    devices=[3],
    max_epochs=30,
    log_every_n_steps=1,
    callbacks=[MyPrintingCallback()],
)

trainer.fit(model, dataloader)
```

    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7,8,9,10,11]
    
      | Name                | Type             | Params
    ---------------------------------------------------------
    0 | FeatureInforEncoder | ModuleList       | 270 K 
    1 | TCPConfidenceLayer  | ModuleList       | 153   
    2 | TCPClassifierLayer  | ModuleList       | 306   
    3 | FeatureEncoder      | ModuleList       | 45.1 K
    4 | MMClasifier         | Sequential       | 302   
    5 | criterion           | CrossEntropyLoss | 0     
    ---------------------------------------------------------
    316 K     Trainable params
    0         Non-trainable params
    316 K     Total params
    1.267     Total estimated model params size (MB)



    Sanity Checking: 0it [00:00, ?it/s]


    Training is starting



    Training: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]



    Validation: 0it [00:00, ?it/s]


    `Trainer.fit` stopped: `max_epochs=30` reached.


    Training is ending


**Test**


```python
trainer.test(model, dataloader)
print(model.test_result)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7,8,9,10,11]



    Testing: 0it [00:00, ?it/s]


    {'test_metric_1': 0.5789473684210527, 'test_metric_2': 0.7241379310344828, 'test_metric_3': 0.5658263305322129}


**Tensorboard Result**

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Pytorch_Light/img4.svg)

### Explain PyTorch Lightning Trainer

Trainer의 작동 방식을 살펴보게 되면 다음과 같다.
1. DataLoader & Model을 정의한다. Trainer의 Input으로 사용된다.
2. Trainer를 정의한다. (Trainer의 자세한 Argument는 나중에 다시 확인하자.)
3. <code>trainer.fit(model, dataloader)</code>로서 Model을 Training하고 Validation한다.
    - 3.1. Training은 LightningModule에서 <code>training_step</code>을 수행한다. <code>training_step</code>은 기존의 PyTorch와 마찬가지로 
        - (1) Input을 받는다. LightningDataModule에서 <code>train_dataloader</code>부분 이다.  
        - (2) Model에서 Forward로서 Output을 구한다. 
        - (3) Loss를 계산한다. 
        - (4) <code>configure_optimizers</code>을 통하여 Model을 학습한다.
    - 3.2. Validation은 LightningModule에서 <code>validation_step</code>과 <code>validation_epoch_end</code>을 수행한다. <code>validation_epoch_end</code>은 기존의 PyTorch와 마찬가지로 
        - (1) Input을 받는다. LightningDataModule에서 <code>val_dataloader</code>부분 이다.  
        - (2) Model에서 Forward로서 Output을 구한다. 
        - (3) Loss를 계산한다. **주요한 점은 <code>self.log("ptl/val_loss", avg_loss)</code>을 통하여 Validation의 Loss를 Log에 저장한다는 것 이다.**
4. <code>trainer.test(model, dataloader)</code>로서 Model을 Test한다. Test의 순서는 다음과 같다.
    - (1) Input을 받는다. LightningDataModule에서 <code>train_dataloader</code>부분 이다.
    - (2) Test결과를 저장한다. 저는 <code>test_dataloader</code>에서 <code>self.test_result = {'test_metric_1': test_metric_1, 'test_metric_2': test_metric_2, 'test_metric_3': test_metric_3}</code>의 Code로서 Test결과를 Model에 저장하였다.
5. Model이 학습이 잘 됬는지 살펴보기 위하여 Log File을 살펴본다. 실제 Log File을 Tensorboard로서 확인하게 되면, <code>self.log("ptl/val_loss", avg_loss)</code>의 값이 저장되어있는 것을 살펴볼 수 있다.



### PyTorch Lightning Trainer Argument
Link: https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/trainer/trainer.py

PyTorch Lightning Trainer에서 지원하는 Argument는 많이 존재한다.
그 중 대표적으로 많이 사용되는 것들의 내용을 살펴보면 다음과 같다.

- <code>accelerator</code>: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto") // Model을 Training하는데 사용할 Device를 설정한다.
- <code>devices</code>: Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`, based on the accelerator type. // Device를 특정하여서 사용할 수 있다. <code>devices=[3]</code>은 gpu devices중에서 3번 device를 사용하겠다는 의미이다.
- <code>gpus</code>: Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node // 몇개의 gpu를 사용할지에 대한 option 이다.
- <code>max_epochs</code>: Stop training once this number of epochs is reached. (default=1000) // Model을 몇번 학습할 지에 대한 option 이다.
- <code>log_every_n_steps</code>: How often to log within steps. (default=50) // Model을 학습할 때 몇번의 epoch마다 Log를 저장할 지에 대한 option 이다.
- <code>callbacks</code>: callbacks: Add a callback or list of callbacks. // Callback으로서 Model을 학습하는 단계마다 어떻게 진행되는지 알기 위하여 사용할 수 있다. <a href="https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.Callback.html#pytorch_lightning.callbacks.Callback">Callback Document</a>에서 어떠한 Event마다 Callback을 사용할 수 있는지 나와있다.
- <code>logger</code>: Logger (or iterable collection of loggers) for experiment tracking. 지정한 Log에 대한 option을 지정할 수 있다. 사용할 수 있는 Log의 종류와 Step or Epoch마다 어떠한 Event(=Callback)의 상황에서 저장할 수 있는지는 <a href="https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html">Logger Document</a>에서 자세히 설명되어있다.

**더 많이 사용되는 Argument는 많이 존재하고, 상황에 따라 맞게 사용되어야 한다. 하지만, 현재 정리되어있는 Document는 없어 매 상황마다 검색해서 사용해야 하는 단점이 있다.**

### Appendix 1. Hyperparameters Load
Link: https://velog.io/@gunwoohan/Pytorch-Lightning-Commom-Use-Cases-04-Hyperparameters

PyTorch Lightning에서 Hyperparameters를 저장하는 방법은 주로 3가지 이다.
1. LightningModule의 <code>__init__</code>에서 <code>self.hyperparameters()</code> method를 사용하는 방법이다. 모든 hyperparameter는 checkpoint파일에 저장된다. (이 방법으로 experiments 진행)


```python
class LitMNIST(LightningModule):
    def __init__(self, layer_1_dim=128, learning_rate=1e-2, **kwargs):
        super().__init__()
        # call this to save (layer_1_dim=128, learning_rate=1e-4) to the checkpoint
        self.save_hyperparameters()

        # equivalent
        self.save_hyperparameters("layer_1_dim", "learning_rate")

        # Now possible to access layer_1_dim from hparams
        self.hparams.layer_1_dim
```

2. 전체다 저장하지 않고 일부만 저장하고 싶을 때는 다음과 같디 <code>save_hyperparameters()</code>를 사용하면 된다.


```python
class LitMNIST(LightningModule):
    def __init__(self, loss_fx, generator_network, layer_1_dim=128 ** kwargs):
        super().__init__()
        self.layer_1_dim = layer_1_dim
        self.loss_fx = loss_fx

        # call this to save (layer_1_dim=128) to the checkpoint
        self.save_hyperparameters("layer_1_dim")
```

3. dict나 namespace 형태도 한번에 hparams에 넘겨줄 수 있음


```python
class LitMNIST(LightningModule):
    def __init__(self, conf: Optional[Union[Dict, Namespace, DictConfig]] = None, **kwargs):
        super().__init__()
        # save the config and any extra arguments
        self.save_hyperparameters(conf)
        self.save_hyperparameters(kwargs)

        self.layer_1 = nn.Linear(28 * 28, self.hparams.layer_1_dim)
        self.layer_2 = nn.Linear(self.hparams.layer_1_dim, self.hparams.layer_2_dim)
        self.layer_3 = nn.Linear(self.hparams.layer_2_dim, 10)
```

저장된 Hyperaprameters는 다음과 같이 쉽게 불러올 수 있다.


```python
import torch
checkpoint = torch.load('./Logs/exp/version_3/checkpoints/epoch=9-step=100.ckpt', map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
```

    {'in_dim': [300, 300, 300], 'num_class': 2, 'binary': True, 'config': {'lr': 1e-05, 'reg': 1e-05, 'hidden_dim': 50}}


또한, 저장된 Checkpoint로서 Model을 불러오기도 다음과 같이 쉽다.


```python
model = MMDynamic.load_from_checkpoint("./Logs/exp/version_3/checkpoints/epoch=9-step=100.ckpt")
print('lr: {}, reg: {}'.format(model.lr, model.reg))
```

    lr: 1e-05, reg: 1e-05


### Appendix 2. Reproducibility

PyTorch Lightning Trainer에서 Reproduction을 위하여 Seed를 고정하기 위해서는 2개의 Option을 지정하여야 한다.

1. <code>pytorch_lightning.seed_everything</code>의 Seed를 고정하여야 한다.
2. <code>pytorch_lightning.Trainer</code>의 <code>deterministic=True</code>를 지정하여야 한다.


```python
from pytorch_lightning import Trainer, seed_everything

seed_everything(42, workers=True)
# sets seeds for numpy, torch and python.random.
model = Model()
trainer = Trainer(deterministic=True)
```

---
layout: post
title:  "PyTorch Lightning Ch5-EarlyStopping"
date:   2023-01-10 11:00:20 +0700
categories: [PytorchLightning]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## EarlyStopping
Link: https://pytorch-lightning.readthedocs.io/en/stable/common/early_stopping.html

Overfitting 방지를 위하여 EarlyStopping을 사용하게 된다. PyTorch Lightning에서는 아래와 같이 Callback을 사용하영 정의할 수 있다.
Early Stopping은 아래와 같이 정의될 수 있다.

1. Import EarlyStopping callback.
2. Log the metric you want to monitor using log() method.
3. Init the callback, and set monitor to the logged metric of your choice.
4. Set the mode based on the metric needs to be monitored.
5. Pass the EarlyStopping callback to the Trainer callbacks flag.

### PyTorch Lightning EarlyStopping Argument
Link: https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/early_stopping.html

PyTorch Lightning EarlyStopping또한 많은 Argument가 존재하게 되며, 아래는 많이 사용되는 Argument에 대한 내용이다.
- <code>monitor</code>: quantity to be monitored // EarlyStopping에 사용할 Metric이다. Accuracy or Loss등으로 정의될 수 있다.
- <code>min_delta</code>: minimum change in the monitored quantity to qualify as an improvement // 특정 값 이하로 값이 변하게 되면, Model의 Training이 수렴했다고 생각하기 위하여 정의되는 값 이다.
- <code>patience</code>: number of checks with no improvement after which training will be stopped. // 얼만큼 기다릴지에 대한 argument이다. 즉, patience=3, min_delta=1e-3이면, 3번동안 monitor하는 metric이 1e-3보다 적으면 EarlyStopping을 적용한다는 것 이다.
- <code>mode</code>: one of 'min', 'max'. In 'min' mode, training will stop when the quantity // 값이 어떻게 변화는지에 초점을 맞추겠다는 것 이다. 대부분 Accuracy와 같이 performance를 측정하는 metric은 값이 클수록 좋으므로, mode=max로 설정하고, Loss와 같은 metric은 mode=min으로서 설정한다.
- <code>verbose</code>: verbosity mode. // Print하여 실제 결과를 확인하는 방법

### EarlyStopping Example

이전에 Training되는 Code에 Early Stopping을 적용하면 아래와 같이 Code를 변경하여 사용하여야 한다.

**Import Library**


```python
from Utils import TCP_DataModule
from MLDRL_Lighting import MMDynamic

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything

# 1. Import EarlyStopping callback.
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
```

**Setting Seed**


```python
seed_everything(42, workers=True)
```

    Global seed set to 42





    42



**DataLoad**


```python
dataloader = TCP_DataModule(data_type='ROSMAP', cv=0, batch_size=32, num_workers=10)
```

**Define Model**


```python
# Model Define
config = {
    "lr": 1e-3,
    "reg": 1e-5,
    "hidden_dim": 50}

model = MMDynamic([300, 300, 300], 2, True, config)
```

**Define Callback**
1. Print Callback
2. **EarlyStopping Callback**

Appendix. EarlyStopping에서 **2. Log the metric you want to monitor using log() method** 과정은 LightningModule에서 <code>self.log("ptl/val_loss", avg_loss)</code>로서 정의하였다.


```python
class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

# 3. Init the callback, and set monitor to the logged metric of your choice.
# 4. Set the mode based on the metric needs to be monitored.
early_stop_callback = EarlyStopping(monitor="ptl/val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
```

**Model Train & Validation**


```python
trainer = Trainer(
    logger=TensorBoardLogger(save_dir="./Logs", name="Early", default_hp_metric=False), # 실험 로거 정의
    accelerator='gpu',
    devices=[4],
    max_epochs=10000,
    log_every_n_steps=1,
    # 5. Pass the EarlyStopping callback to the Trainer callbacks flag.
    callbacks=[MyPrintingCallback(), early_stop_callback],
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


    Training is ending


**Test**


```python
trainer.test(model, dataloader)
print(model.test_result)
```

    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3,4,5,6,7,8,9,10,11]



    Testing: 0it [00:00, ?it/s]


    {'test_metric_1': 0.6842105263157895, 'test_metric_2': 0.6666666666666666, 'test_metric_3': 0.8151260504201681}


**Tensorboard Result**

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Pytorch_Light/img5.svg)

실제 결과를 확인해보면, max_epochs=10000으로 설정하였지만, EarlyStopping때문에 Epoch 47에서 학습을 멈추는 것을 알 수 있다.

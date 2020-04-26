---
layout: post
title:  "CuDNN Error-No Initialization"
date:   2020-04-05 12:00:20 +0700
categories: [others]
---


### CuDNN Error- No Initialization
Tensorflow-GPU Version을 새로운 Computer의 환경에 맞게 Setting하다 보니 많은 CuDNN쪽에서 Error가 발생하여서 정리하였다.  

먼저, CUDA, CUDNN, Tensorflow를 정상적으로 잘 설치하였다고 가정하였을 때, Tensorflow작동 시 다음과 같은 Error가 발생할 수 있다.  
```code
tensorflow.python.framework.errors_impl.UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
	 [[node sequential/conv2d/Conv2D (defined at ./a.py:30) ]] [Op:__inference_distributed_function_886]

Function call stack:
distributed_function
```
<br>

이러한 Erorr는 **CuDNN이 잘 설치되었다는 가정 하에서 그래픽 카드의 메모리를 초과하기 때문에 발생하는 Error이다.** Nvidia-SMI 명령어로 Graphic Card의 사용량을 확인하면 보면 초과하거나 꽉 차있을 확률이 높다. 이럴때 해결하는 방안이 몇가지가 있다.

1. Batch의 수를 줄여라: 가장 기본적인 방법이지만, 지속적으로 Batch의 개수를 줄여가면서 지속적으로 확인하는 방법이다.

2. Tensorflow내에서 Graphic Card의 Memory의 제한을 걸어라.  

기존 까지는 1의 방법대로 돌아갈때까지 지속적으로 조금씩 낮췄지만, Code를 여러개를 돌리거나, 혹은 좀 더 정확하게 설계하는 방법은 다음과 같다.  
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
```
<br>
위의 Code와 같은 방법으로서 직접적으로 Memory를 제한 할 수 가 있고 이에 자신의 그래픽 카드의 성능에 맞춰서 설계를 할 수 있다.  


<br>

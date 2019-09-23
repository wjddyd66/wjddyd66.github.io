---
layout: post
title:  "Pytorch-Basic"
date:   2019-09-23 09:00:00 +0700
categories: [Pytorch]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
### Pytorch vs Tensorflow
기존 DeepLearning을 Tensorflow로서 작성하였으나 Pytorch라는 새로운 Framework를 배우기 위하여 2 FrameWork간의 차이를 살펴보자.  
참조: <a href="https://wjddyd66.github.io/category/Tensorflow">Tensorflow 자세한 내용</a><br>
Pytorch와 Tensorflow의 가장 큰 차이점은 작동 방싱에서 있다. 아래 그림을 살펴보자.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/9974C3485C3085C80C" height="250" width="600" /></div>
위의 그림을 살펴보게 되면 다음과 같은 차이가 있다.  
**Tensorflow: Static Graph**  
- 매 iteration 단계에서 기존에 구축된 정적인 동일한 computational graph에 sess.run방식으로 동작(FP)  
- 동일한 그래프를 반복적으로 다시 사용하게 되므로 framework에서 그래프에 대한 최적화를 진행
- 그래프의 자료구조를 disk로 serialize를 할 수 있다. 이로 인해 original code에 대한 Access없이 해당 파일만으로 이용하여 구동 가능. 즉, Platform이 서로 다른 언어에서 쉽게 네트워크를 import가능(.proto 사용)<br><a href="https://wjddyd66.github.io/others/2019/08/21/GoogleProtocolBuffers.html">Google Protocol Buffer3 자세한 내용</a>


**Pytorch: Dynamic Graph**  
- 매 iteration을 통해 수행되는 FP에서 새로운 computational graph를 생성
- Code가 깔끔해지고 쉬워짐

실질적인 비교를 위해서 Code로서 Pytorch 설치부터 Python, Tensorflow, Pytorch의 구동시간을 각각 비교해보자.  

<br><br>

### Installation
- 파이썬 버젼 체크 (Python version Check)
- 파이토치 설치 (PyTorch Installation)
- 쿠다 및 CuDNN 체크 (Cuda & CuDNN Check)

#### 1. Python Version Check
파이썬 버젼 체크

```python
import sys
print(sys.version)
```
```code
3.6.8 (default, Jan 14 2019, 11:02:34) 
[GCC 8.0.1 20180414 (experimental) [trunk revision 259383]]
```

#### 2. PyTorch Installation
- 구글 콜라브 버젼에 따라 파이토치가 설치되어 있을수도 있고 아닐 수도 있습니다.
- 설치가 안되어 있을 경우 아래와 같은 명령어로 설치하면 됩니다.
- !pip3 install torch torchvision

```python
import torch
!pip3 install torch torchvision
```
```code
Collecting torch
  Using cached https://files.pythonhosted.org/packages/30/57/d5cceb0799c06733eefce80c395459f28970ebb9e896846ce96ab579a3f1/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
Collecting torchvision
  Using cached https://files.pythonhosted.org/packages/06/e6/a564eba563f7ff53aa7318ff6aaa5bd8385cbda39ed55ba471e95af27d19/torchvision-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
Collecting numpy (from torch)
  Using cached https://files.pythonhosted.org/packages/e5/e6/c3fdc53aed9fa19d6ff3abf97dfad768ae3afce1b7431f7500000816bda5/numpy-1.17.2-cp36-cp36m-manylinux1_x86_64.whl
Collecting six (from torchvision)
  Using cached https://files.pythonhosted.org/packages/73/fb/00a976f728d0d1fecfe898238ce23f502a721c0ac0ecfedb80e0d88c64e9/six-1.12.0-py2.py3-none-any.whl
Collecting pillow>=4.1.1 (from torchvision)
  Using cached https://files.pythonhosted.org/packages/14/41/db6dec65ddbc176a59b89485e8cc136a433ed9c6397b6bfe2cd38412051e/Pillow-6.1.0-cp36-cp36m-manylinux1_x86_64.whl
Installing collected packages: numpy, torch, six, pillow, torchvision
Successfully installed numpy-1.17.2 pillow-6.1.0 six-1.12.0 torch-1.2.0 torchvision-0.4.0
```

####  3. Cuda & cudnn Version Check
- 파이토치를 통해 각각 몇 버젼이 설치 되어있는지 확인해줍니다.

```python
import torch

print("Torch version:{}".format(torch.__version__))
print("cuda version: {}".format(torch.version.cuda))
print("cudnn version:{}".format(torch.backends.cudnn.version()))
```
```code
Torch version:1.2.0
cuda version: 10.0.130
cudnn version:7602
```

#### 4. PyTorch CPU & GPU Tensor Check
- 파이토치 텐서를 생성해봄으로써 제대로 설치 되었는지, 잘 동작하는지 확인해줍니다.

##### 4-1 Create CPU tensor

```python
# 0으로 차있는 2x3 형태의 텐서를 생성합니다.
cpu_tensor = torch.zeros(2,3)
print(cpu_tensor)
```
```code
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

##### 4-2 Allocate tensor on GPU

```python
# 어느 장치(cpu 혹은 gpu)에 텐서를 올릴지 지정합니다.
# 아래는 torch.device라는 함수를 사용해 gpu로 장치를 지정합니다. 
device = torch.device('cuda')

# gpu가 사용 가능한지 확인해줍니다.
if torch.cuda.is_available():
  
  # https://pytorch.org/docs/stable/tensors.html?highlight=#torch.Tensor.to
  # cpu에 있었던 텐서를 to 함수를 이용해 지정해놓은 장치(여기서는 gpu)로 올려줍니다.
  gpu_tensor = cpu_tensor.to(device)
  print(gpu_tensor)
```
```code
tensor([[0., 0., 0.],
        [0., 0., 0.]], device='cuda:0')
```

##### 4-3 Reallocate tensor back on CPU
```python
# device 함수와 to 함수를 이용해 gpu에 있던 텐서를 다시 cpu로 옮겨올 수 있습니다.
cpu_tensor_back = gpu_tensor.to(torch.device('cpu'))
cpu_tensor_back
```
```code
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```
<br><br>
### Framework Comparison

- Python vs Tensorflow vs PyTorch
- 같은 연산이 각각 어떻게 구동이 되는지 알아보고 속도 역시 비교해보도록 하겠습니다.
- x * y + z

##### Python
```python
# 연산에 필요한 numpy, 시간을 측정하기 위해 datetime을 불러옵니다.
import numpy as np 
from datetime import datetime
start = datetime.now()

# 랜덤하게 3x4 형태의 변수 x,y,z를 설정해줍니다.
np.random.seed(0)

N,D = 3,4

x = np.random.randn(N,D)
y = np.random.randn(N,D)
z = np.random.randn(N,D)

# x,y,z를 이용해 x*y+z를 계산해줍니다.
a = x * y
b = a + z
c = np.sum(b)

# 기울기(gradient)가 1이라고 가정하고 역전파를 해줍니다. 역전파에 대한 내용은 4장에서 자세히 다룹니다.
grad_c = 1.0
grad_b = grad_c * np.ones((N,D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_y = grad_a * y
grad_x = grad_a * x

# 각각의 기울기가 몇인지 걸린 시간은 얼마인지 확인해봅니다.
print(grad_x)
print(grad_y)
print(grad_z)
print(datetime.now()-start)
```
```code
[[ 1.76405235  0.40015721  0.97873798  2.2408932 ]
 [ 1.86755799 -0.97727788  0.95008842 -0.15135721]
 [-0.10321885  0.4105985   0.14404357  1.45427351]]
[[ 0.76103773  0.12167502  0.44386323  0.33367433]
 [ 1.49407907 -0.20515826  0.3130677  -0.85409574]
 [-2.55298982  0.6536186   0.8644362  -0.74216502]]
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
0:00:00.001167
```

#####  Tensorflow

```python
# 이번에는 텐서플로 프레임워크를 이용해 같은 연산을 해보도록 하겠습니다.
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import numpy as np
from datetime import datetime
start = datetime.now()

# 텐서플로는 연산 그래프를 먼저 정의하고 추후에 여기에 값을 전달하는 방식입니다. 여기서는 비어있는 그래프만 정의해줍니다.
# Define Graph on GPU
x = tf.placeholder(tf.float32)     # 비어있는 노드인 placeholder를 정의하고 여기에 들어가는 데이터타입을 명시 해놓습니다.
y = tf.placeholder(tf.float32)
z = tf.placeholder(tf.float32)

a = x * y                          # 연산 과정 또한 정의해줍니다.
b = a + z
c = tf.reduce_sum(b)
    
grad_x, grad_y, grad_z = tf.gradients(c,[x,y,z])  # c에 대한 x,y,z의 기울기(gradient)를 구하고 이를 각각 grad_x, grad_y, grad_z에 저장하도록 지정해놓습니다.

# 실제적인 계산이 이루어지는 부분. 텐서플로에서는 이를 세션이라고 합니다.
with tf.Session() as sess:
    values = {
        x: np.random.randn(N,D),     # 여기서 실제 값들이 생성됩니다.
        y: np.random.randn(N,D),
        z: np.random.randn(N,D)           
    }
    out = sess.run([c,grad_x,grad_y,grad_z],feed_dict = values)  # 세션에서 실제로 값을 계산하는 부분입니다. feed_dict를 통해서 값들을 전달합니다.
    c_val, grad_x_val, grad_y_val, grad_z_val = out

# 값들을 확인하고 걸린 시간을 측정합니다.
print(grad_x_val)
print(grad_y_val)
print(grad_z_val)
print(datetime.now()-start)
```
```code
[[-1.6138978  -0.21274029 -0.89546657  0.3869025 ]
 [-0.51080513 -1.1806322  -0.02818223  0.42833188]
 [ 0.06651722  0.3024719  -0.6343221  -0.36274117]]
[[ 1.2302907   1.2023798  -0.3873268  -0.30230275]
 [-1.048553   -1.420018   -1.7062702   1.9507754 ]
 [-0.5096522  -0.4380743  -1.2527953   0.7774904 ]]
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
0:00:00.061126
```

#####  Pytorch
```python
# 이번에는 파이토치를 이용해 같은 연산을 진행해보도록 하겠습니다.
import torch
from datetime import datetime
start = datetime.now()

N,D = 3,4

# x,y,z를 랜덤하게 초기화 해줍니다. 
# https://pytorch.org/docs/stable/torch.html?highlight=randn#torch.randn
x = torch.randn(N,D,device=torch.device('cuda'), requires_grad=True)
y = torch.randn(N,D,device=torch.device('cuda'), requires_grad=True)
z = torch.randn(N,D,device=torch.device('cuda'), requires_grad=True)

# 연산 그래프는 정의됨과 동시에 연산됩니다.
a = x * y
b = a + z
c = torch.sum(b)

# 기울기(gradient)가 1.0라고 가정하고 최종 값인 c에서 backward를 통해 역전파를 해줍니다.
# 넘파이와 비교했을때 이 과정이 자동적으로 게산되는 것을 확인할 수 있습니다.
c.backward(gradient=torch.cuda.FloatTensor([1.0]))

# 각각의 기울기와 걸린 시간을 출력합니다.
print(x.grad)
print(y.grad)
print(z.grad)
print(datetime.now()-start)
```
```code
tensor([[ 8.2338e-01, -5.6320e-01, -2.3355e-03,  3.7213e-01],
        [ 1.0584e-01, -9.1745e-01,  1.8612e+00,  8.3635e-01],
        [-1.9306e+00,  1.1427e+00,  7.5820e-01, -3.4195e-04]], device='cuda:0')
tensor([[-2.4330, -0.3690, -0.5224,  0.4084],
        [-0.3373,  0.7050,  0.7853,  0.2445],
        [-0.5182, -1.0441,  0.5183, -0.5510]], device='cuda:0')
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]], device='cuda:0')
0:00:00.003761
```

간단한 예제를 통하여 작동시간은 **Python < Pytorch < Tensorflow**를 확인 가능하다.  
<br><br>
### Pytorch 기본 동작 방법
X라는 변수에 파이토치 텐서를 하나 생성해서 지정, shape = (2,3)  
텐서에는 임의의 난수가 들어간다.
```python
X = torch.Tensor(2,3)
print(X)
```
```code
tensor([[-3.4702e-01,  4.5702e-41, -3.4702e-01],
        [ 4.5702e-41,  1.3563e-19,  4.5071e+16]])
```
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

텐서를 생성하면서 원하는 값으로 초기화 하려면 인수로 배열을 전달  

torch.tensor함수는 인수로 data, dtype, device, requires_grad를 전달
- data: tensor에 넣을 data => 배열로 전달
- dtype: data의 타입 결정
- requres_grad: 텐서에 대한 기울기를 저장할 지 여부 지정
- device: 텐서를 어느 기기에 올릴 것인지 명시

**dtype 종류**
<table class="docutils align-default">
<colgroup>
<col style="width: 19%">
<col style="width: 34%">
<col style="width: 21%">
<col style="width: 25%">
</colgroup>
<thead>
<tr class="row-odd"><th class="head"><p>Data type</p></th>
<th class="head"><p>dtype</p></th>
<th class="head"><p>CPU tensor</p></th>
<th class="head"><p>GPU tensor</p></th>
</tr>
</thead>
<tbody>
<tr class="row-even"><td><p>32-bit floating point</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.float32</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.float</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.FloatTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.FloatTensor</span></code></p></td>
</tr>
<tr class="row-odd"><td><p>64-bit floating point</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.float64</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.double</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.DoubleTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.DoubleTensor</span></code></p></td>
</tr>
<tr class="row-even"><td><p>16-bit floating point</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.float16</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.half</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.HalfTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.HalfTensor</span></code></p></td>
</tr>
<tr class="row-odd"><td><p>8-bit integer (unsigned)</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.uint8</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.ByteTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.ByteTensor</span></code></p></td>
</tr>
<tr class="row-even"><td><p>8-bit integer (signed)</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.int8</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.CharTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.CharTensor</span></code></p></td>
</tr>
<tr class="row-odd"><td><p>16-bit integer (signed)</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.int16</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.short</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.ShortTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.ShortTensor</span></code></p></td>
</tr>
<tr class="row-even"><td><p>32-bit integer (signed)</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.int32</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.int</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.IntTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.IntTensor</span></code></p></td>
</tr>
<tr class="row-odd"><td><p>64-bit integer (signed)</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.int64</span></code> or <code class="docutils literal notranslate"><span class="pre">torch.long</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.LongTensor</span></code></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.LongTensor</span></code></p></td>
</tr>
<tr class="row-even"><td><p>Boolean</p></td>
<td><p><code class="docutils literal notranslate"><span class="pre">torch.bool</span></code></p></td>
<td><p><a class="reference internal has-code" href="#torch.BoolTensor" title="torch.BoolTensor"><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.BoolTensor</span></code></a></p></td>
<td><p><code class="xref py py-class docutils literal notranslate"><span class="pre">torch.cuda.BoolTensor</span></code></p></td>
</tr>
</tbody>
</table>
```python
# 텐서에 직접 값 대입
X = torch.tensor([[1,2,3],[4,5,6]])
print(X)
```

```code
tensor([[1, 2, 3],
        [4, 5, 6]])
```
연산 그래프 생성후 Gradient Descent 로 기울기 확인  
x tensor에만 requires_grad = True로 설정하여 값 확인 가능
```python
x = torch.tensor(data=[2.0,3.0],requires_grad=True)
y = x**2
z = 2*y + 3

target = torch.tensor([3.0, 4.0])
loss = torch.sum(torch.abs(z-target))
loss.backward()

print(x.grad, y.grad, z.grad)
```
```code
tensor([ 8., 12.]) None None
```
앞으로의 Pytorch Post는 실제 구현만 올린다.  
Pytorch로서 구현하는 실질적인 개념은 <a href="https://wjddyd66.github.io/category/DL">DeepLearning 개념</a><br>을 참조
<br>

<hr>
참조: <a href="https://github.com/wjddyd66/Pytorch/blob/master/Pytorch_vs_Tensorflow.ipynb">원본코드</a> <br>
참조: <a href="https://dev-jm.tistory.com/4">dev-jm 블로그</a> <br>
참조: 파이토치 첫걸음<br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
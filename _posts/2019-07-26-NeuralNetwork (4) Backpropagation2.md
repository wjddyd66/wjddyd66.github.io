---
layout: post
title:  "NeuralNetwork (4) Backpropagation2"
date:   2019-07-26 11:40:00 +0700
categories: [DL]
---

### Backpropagation
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
이번 Post 에서는 저번 Post에서 공부하였던 Backpropagation을 간단한 것 부터 많이 사용하는 것 까지 하나하나 구현해보며 실제로는 어떻게 Code를 작성해야 하는지 알아보자.  

#### 덧셈 노드의 역전파
덧셈 노드의 역전파는 입력값을 그대로 흘려보낸다. 이를 보고 gradient distributor라고 한다.  
<p>$$z = x+y$$</p>
<p>$$\frac{\partial z}{\partial x} = 1$$</p>
<p>$$\frac{\partial z}{\partial y} = 1$$</p>
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile24.uf.tistory.com%2Fimage%2F99FB57455B98F67407FADD" height="200" width="600" />
위의 <span>$$z = x + y$$</span>계산은 전체 그래프의 중간 어딘가에 존재한다고 가정했기 때문에, 이 계산 그래프의 앞부분에서 부터 <span>\frac{\partial L}{\partial z}</span>가 전해졌다고 가정한다.  
위의 그림은 아래와 같은 Code로서 간단히 구현될 수 있다.  
```python
class AddLayr:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
    #dout은 알에서 전해지는 값 이다.
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
```

#### 곱셈 노드의 역전파
곱셈 노드의 역전파는 입력값의 위치를 서로 바꾼 다음 곱해서 흘려보낸다. 이를 보고 gradient switcher부른다.  
<p>$$z = xy$$</p>
<p>$$\frac{\partial z}{\partial x} = y$$</p>
<p>$$\frac{\partial z}{\partial y} = x$$</p>
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile1.uf.tistory.com%2Fimage%2F99E3EF435B98F69309175D" height="200" width="600" />
위의 <span>$$z = x + y$$</span>계산은 전체 그래프의 중간 어딘가에 존재한다고 가정했기 때문에, 이 계산 그래프의 앞부분에서 부터 <span>\frac{\partial L}{\partial z}</span>가 전해졌다고 가정한다.  
위의 그림은 아래와 같은 Code로서 간단히 구현될 수 있다.  
```python
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx,dy
```
**간단한 신경망 구성**  
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile3.uf.tistory.com%2Fimage%2F99499E4E5B98F6C10EBEA5" height="200" width="600" /></div>
위와 같은 그림으로서 간단한 신경망이 구성되어있을때  
AddLayer 와 MulLayer를 활용하여 구성하게 되면 아래와 같다.  
```python
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print('%d' % price)
print("%d, %.1f, %.1f, %d, %d" % (dapple_num, dapple, dorange, dorange_num, dtax))
```
**결과**  
```code
715
110, 2.2, 3.3, 165, 650
```

#### Activation Function 계층 구현하기
Activation Function에 대한 사전지식은 아래 링크를 참조  
<a href="https://wjddyd66.github.io/dl/2019/07/26/NeuralNetwork-(1)-Basic-&-Activation-Function.html">Activation Function 자세한 내용</a>
위의 내용에서는 Activation Function에 대한 개념과 식 그리고 미분 방법에 대하여 Post하였다.  
이를 활용하여 Activation Function에 Forward 와 Backward를 실제 구현해보자.  

**ReLU**  
식: <span> $f(x) = max(0,x)$ </span><br>
미분식:

- x > 0 : 1
- x < 0 : 0

순전파 때의 입력인 x가 0보다 크면 역전파는 상류의 값을 그대로 전달하지만, 순전파 때 x가 0이하면 역전파 때는 하류로 신호를 보내지 않는다.  
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile28.uf.tistory.com%2Fimage%2F99E517485B98F6E504DB20" height="200" width="600" /></div>
위의 그림은 아래와 같이 간단한 Code로서 구현 될 수 있다.  
```python
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
```
mask는 True/False로 구선된 Numpy배열로, 순전파의 입력인 x의 원소값이 0이하인 Index는 True, 그 외는 False로 유지한다.  

**Sigmoid**  
식: <span> $\sigma(x) = {1 \over 1+e^{-x}}$ </span><br>
미분식: <span> $\sigma\prime(x) = \sigma(x)(1-\sigma(x))$ </span>  

Sigmoid의 경우 ReLu보다 식이 복잡하여 아래그림과 같이 Sigmoid계산 과정을 쭉 펼쳐서 생각해 보자.
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile28.uf.tistory.com%2Fimage%2F99E517485B98F6E504DB20" height="200" width="600" /></div>

**1. / 과정**  
<p>$$\frac{\partial y}{\partial x} = -\frac{1}{x^2} = -y^2$$</p>
역전파 때 상류에서 흘러온 값에 제곱 후 - 를 곱하여 보낸다.  
**2. + 과정**  
+의 경우 위에서 증명하였듯이 그냥 흘려보낸다.  
**3. exp 과정**  
<p>$$\frac{\partial y}{\partial x} = exp(x)$$</p>
exp(x)는 미분하여도 값이 똑같다.  
상류에서 흘러온 값에 exp(x)를 곱하여 흘려보내 준다.  
**4. x 과정**  
x 의 경우 위에서 증명하였듯이 입력값의 위치를 서로 바꾼 다음 곱해서 흘려 보낸다.  

최종적인 식을 정리하면 아래와 같다.  
<p>$$\frac{\partial L}{\partial y} y^2 exp(-x)$$</p>
<p>$$= \frac{\partial L}{\partial y} \frac{1}{(1+exp(-x))^2} exp(-x)$$</p>
<p>$$= \frac{\partial L}{\partial y} \frac{1}{1+exp(-x)} \frac{exp(-x)}{1+exp(-x)}$$</p>
<p>$$= \frac{\partial L}{\partial y} y(1-y)$$</p>
위의 식에서 알 수 있듯이 Sigmoid 계층의 역전파는 순전파의 출력(y)만으로 계산할 수 있다.  

위의 그림은 아래와 같이 간단한 Code로서 구현 될 수 있다.  
```python
class Relu:
    def __init__(self):
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
```



<hr>
참조: <a href="https://excelsior-cjh.tistory.com/171">excelsior-cjh 블로그</a> <br>
참조: 밑바닥부터 시작하는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
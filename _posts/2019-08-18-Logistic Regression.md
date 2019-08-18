---
layout: post
title:  "Tensorflow-Logistic Regression"
date:   2019-08-18 10:00:00 +0700
categories: [Tensorflow]
---

### Logistic Regression
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

로지스틱 회귀분석은 예측하는 Linear Regression과 달리 Y가 범주형일때 사용하게 된다.  
로지스틱 회귀는 이항형 또는 다향형이 될 수 있다. 종속변수의 결과가 2개의 종류라면 이항형, 그 이상이라면 다항형이다.  
이항형 다항형에 따라 활성화 함수(active function)이 아래와 같은 종류를 가지게 된다.  
<span style ="color: red">**이항형: 시그모이드 혹은 하이퍼 볼릭 탄젠트, 다항형: 소프트맥스**</span><br>
<a href="https://wjddyd66.github.io/r/2019/06/17/Regression.html">로지스틱 회귀분석 자세한 내용</a><br>

로지스틱 회귀분석에서 활성화 함수를 시그모이드를 사용한다고 하면 아래와 같은 식을 얻을 수 있다.  
<p> $$ y(x) = {1 \over 1+e^{-ax+b}}$$ </p><br>
위의 식에서 a값을 변화시켰을 경우 그래프의 기울기를 변화시킬 수 있고, b값을 변화시켜 그래프를 평행이동 시킬 수 있다.  
a값을 변화시켰을때의 시그모이드 그래프  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/5.png" height="250" width="600" /></div>
b값을 변화시켰을때의 시그모이드 그래프  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/6.png" height="250" width="600" /></div>

### Cross Entropy
Logistic Regression또한 Linear Regression과 같이 MSE를 사용하여 구할 수 있으나 Cross Entropy를 사용하여 Model의 성능을 향상시킬 수 있다.  
Cross Entropy를 이해하기 위해서 <span style ="color: red">**정보량, 엔트로피**</span>의 개념을 알고 있어야 한다.  
<br>
<span style ="color: red">**정보량**</span>은 아래와 같은 식으로 표현할 수 있다.  
<p>$$I(x) = log(\frac{1}{p(x)}) $$ </p><br>
<span>$$(1 \over p(x)) $$ </span>은 사건이 발생할 수 있는 확률이다.  
이러한 값에 log를 취함으로 인하여 <span style ="color: red">**필요한 최소한의 자원**</span>을 나타낸다.  
<br>
<span style ="color: red">**Entropy**</span>는 아래와 같은 식으로 표현할 수 있다.  
<p>$$H_p(X)=\sum_{i=0}^n  p(x_i)log(p(x_i)) $$ </p><br>
Entropy는 <span style ="color: red">**정보량에 대한 기댓값이며 동시에 사건을 표현하기 위해 요구되는 평균 자원이라고 할 수 있다.**</span>으로 정의된다.  
예측이 어려울수록 정보의 양은 더 많아지고 엔트로피는 더 커진다.  
<br>
<span style ="color: red">**Cross Entropy**</span>의식은 아래와 같다.  
<p>$$H_{p,q}(x)=\sum_{i=0}^n  p(x_i)log(q(x_i)) $$ </p><br>
Entropy는 <span style ="color: red">**p는 true label에 대한 분포를, q는 현재 예측모델의 추정값에 대한 분포**</span>를 의미하게 된다.  
위의 식은 아래와 같은 식으로 나타낼 수 있다.  
<p>$$f_c(x)=y\prime log(y) - (1-y\prime)log(1-y) $$ </p><br>
위의 식에서 우변의 값을 각각 따로 그래프로 그려보게 되면 아래와 같다.  
<span>$$y\prime log(y)$$ </span>그래프  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/7.png" height="250" width="600" /></div>
<span>$$(1-y\prime)log(1-y)$$ </span>그래프  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/8.png" height="250" width="600" /></div>
<span> $$ y(x) = {1 \over 1+e^{-ax+b}}$$ </span>식에서 a, b값을 조정함에 따라 위의 그래프들의 값이 아래 그림과 같이 변경된다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/9.PNG" height="250" width="600" /></div>
서로 값들이 상봔되는 관계를 가지고 있고 두개의 값을 더했을때 가장 작은 값을 찾는것을 목표로 한다.  
<br>
b에 따른 Cost Function의 식은 아래와 같다.  
<p>$$b(update)=b-\alpha\frac{\partial f_c(x)}{\partial b}$$</p><br>
위의 식을 그래프로 표현하면 아래와 같이 된다고 생각할 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/10.PNG" height="250" width="600" /></div>
a에 따른 Cost Function의 식은 아래와 같다.  
<p>$$a(update)=a-\alpha\frac{\partial f_c(x)}{\partial a}$$</p><br>
위의 식을 그래프로 표현하면 아래와 같이 된다고 생각할 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/11.PNG" height="250" width="600" /></div>
<br>

### Softmax-with-Loss계층
**Softmax-with-Loss**란 Softmax와 Cross Entropy를 합친 계층이라고 할 수 있다. 자세한 **Softmax-with-Loss**를 알아보기 전에 **Loss Function**의 사용하는 시기에 대해서 정의해보자.  
**Loss Function**을 사용하는 시기  
1. 회귀 분석: MSE
2. 분류 분석: Corss Entropy

**분류 분석: Cross Entropy**에서 **Activation Function**의 종류를 사용하는 시기에 대해서 정의해보자.  
1. 이항 분류 분석: Sigmoid
2. 다항 분류 분석: Softmax

즉 **Softmax-with-Loss**계층은 **다항 분류 분석**에서 사용하는 계층이라는 것을 알 수 있다.  

**Softmax-with-Loss**를 자세히 알아보기 전에 **Softmax-with-Loss**의 대한 **Parameter**를 아래와 같이 정의하고 시작하자.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">

	<tr>	
		<td>Parameter</td><td>의미</td>
	</tr>
	
	<tr>	
		<td>$$n$$</td><td>분류해야할 범주 수</td>
	</tr>
	
	<tr>	
		<td>$$a_i$$</td><td>Softmax의 i 번째 입력값</td>
	</tr>
	
	<tr>	
		<td>$$p_i$$</td><td>Softmax의 i 번째 출력값</td>
	</tr>
	

</table>
또한 앞으로의 내용은 **Softmax 의 미분**에 대해서는 생략 되어있으므로 아래 링크에서 선행 학습이 필요하다.  
<a href="https://wjddyd66.github.io/dl/2019/07/26/NeuralNetwork-(2)-Loss-Function.html">**Softmax 자세한 내용**</a>  
<br>
**Softmax-with-Loss 순전파**  
- Softmax 출력값  
<p>$$p_i = \frac{exp(a_i)}{\sum_n exp(a_n)}$$</p>
- Cross Entropy Loss(L) 값  
<p>$$L = -\sum_j y_j logp_j (y_j:Output값)$$</p>  


**Softmax-with-Loss 역전파**  
<p>$$\frac{\partial L}{\partial a_i} = \frac{\partial (-\sum_j y_j logp_j)}{\partial a_i}$$</p>
<p>$$ = -\sum_j y_j \frac{\partial logp_j}{\partial a_i}$$</p>
<p>$$ = -\sum_j y_j \frac{1}{p_j} \frac{\partial p_j}{\partial a_i}$$</p>
<p>$$ = -\frac{y_i}{p_i}p_i(1-p_i) - \sum_{i \neq j}\frac{y_j}{p_j}(-p_ip_j)$$</p>
<p>$$ = -y_i + y_ip_i + \sum_{i \neq j}y_jp_i$$</p>
<p>$$= - y_i + p_i\sum_j y_j$$</p>
<p>$$p_i - y_i \text{    }\because(\sum_j y_j = 1)$$ </p>  

<br>
**즉 Softmax-with-Loss 노드의 Gradient를 구하려면 입력 벡터에 Softmax를 취한 뒤, 정답 레이블에 해당하는 요소값만 1을 빼주면 된다.**  
**Softmax-with-Loss의 장점**  
1. Gradient를 구하기 쉽다.
2. Gradient가 0으로 죽는 일이 거의 없다.

### Logistic Regression 구현
**Logistic Regression** 또한 앞선 Post에서 다룬 내용 **Linear Regression**과 같이 **MNIST Data**를 분리하는 예제를 사용한다.  
이번 **MNIST Data**의 경우 2가지로 나뉜다.  
2가지 경우에 대한 설명과 사용기법은 아래와 같다.  
<table class="table">

	<tr>	
		<td>사용 기법</td><td>One-Hot-Encodint O</td><td>One-Hot-Encodint X</td>
	</tr>
	
	<tr>	
		<td>Loss Function</td><td>Cross Entropy</td><td>Softmax-with-Loss</td>
	</tr>
	
	<tr>	
		<td>Optimazation</td><td>Gradient Descent</td><td>Gradient Descent</td>
	</tr>
	

</table>
<br>
**One-Hot-Encoding O**  
Tensorflow import
```python
import tensorflow as tf
```
MNIST Data Download One-Hot-Encoding O
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)
```
- x: Input Data, Placeholder, Mnist Image flattening => 28 * 28 = 784 shape=[None,784]
- y: Output Data, Placeholder, 1~9 까지의 숫자 판별이므로 shape=[None,10]

**flattening**: 차원을 1차원으로 바꿔주는 것
```python
x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32,shape=[None,10])
```
- Model: y = Wx + b
- W: weight, Variable, shape = [784,10]
- b: bias, Variable, shape = [10]

```python
W = tf.Variable(tf.zeros(shape=[784,10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf.matmul(x,W)+b
y_ = tf.nn.softmax(logits)
```
- Loss Function: Cross-Entropy
- Optimization: GradientDescent

```python
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
```

Session 및 변수 초기값 할당
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```
Mnist Data 100개씩 불러와서 Mini-Batch 처리
```python
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
```
Model의 정확도 측정 & Session 닫기
```python
correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('정확도(Accuracy): %f'%sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels}))

sess.close()
```
**정확도(Accuracy): 0.916800**  
<br><br>


**One-Hot-Encodint X(Label 사용)**  
MNIST Data Download One-Hot-Encoding X
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=False)
```

y가 One-Hot-Encoding이아닌 Label이므로 2차원이 아닌 1차원으로 선언
```python
x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.int64,shape=[None])
W = tf.Variable(tf.zeros(shape=[784,10]))
b = tf.Variable(tf.zeros(shape=[10]))
logits = tf.matmul(x,W)+b
y_ = tf.nn.softmax(logits)
```
Loss Function으로 softmax_cross_entropy사용  
**주의 사항: Label은 항상 int형만 가능하다.**
```python
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
```
y가 Label 형태의 Scalar이므로 tf.argmax 사용 X
```python
correct_prediction = tf.equal(tf.argmax(y_,1),y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print('정확도(Accuracy): %f'%sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels}))

sess.close()
```
**정확도(Accuracy): 0.918800**
<br><br>
<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow/blob/master/LogisticRegression.ipynb">원본코드</a> <br>
참조: 텐서플로로 배우는 딥러닝<br>
참조: <a href="https://www.youtube.com/watch?v=kHLqMsN7yao&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=23">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="https://curt-park.github.io/2018-09-19/loss-cross-entropy/">curt-park 블로그</a> <br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
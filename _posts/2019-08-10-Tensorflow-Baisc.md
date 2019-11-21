---
layout: post
title:  "Tensorflow-Basic"
date:   2019-08-10 10:00:00 +0700
categories: [Tensorflow]
---

### Tensorflow
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
이전시간에 배운 Perceptron을 실질적으로 구현하기 앞서 알아야할 Tensorflow의 기본적인 지식을 적어둔다.  
Tensorflow는 다음 두 단계로서 이루어 진다.  
1. 수식을 그래프로 포현
2. 그래프를 실행

**텐서플로우는 수치계산을 위한 라이브러리 이다. 데이터 플로우 그래프를 사용하여 복잡한 연산을 단순한 연산으로 나누어서 계산을 하게 된다.**  
Tensorflow의 이러한 특징인 Graph를 노드와 에지로서 사용하여 표현하게 된다.  
Graph에서의 노드를 오퍼레이션(Operation) 즉, 값이나 연산자를 의미하고 값을 전달하는 역할을 한다.  
Graph에서의 에지를 텐서(Tensor) Operation의 출력을 가리키는 심볼릭 링크일 뿐 실제 값을 가지고 있지 않는다.  
Tensor에 실질적인 값을 가지기 위해서는 Session에서 Operation을 실행해야 Tensor는 값을 가지게 된다.  
**TensorFlow는  이러한 복잡한 연산을 Operation 과 Tensor를 이용하여 여려단계의 간단한 오퍼레이션으로 구성하게 된다.**  
이러한 Tensorflow는 아래와 같은 code로서 import가 가능하다.  
```python
import tensorflow as tf
```
<br>

### Tensorflow 자료형
이전까지 배웠던 Python, Java와 다르게 Tensorflow는 3가지의 자료형을 가지고 있다.  
상수형(Constant), 플레이스 홀더(Placeholder), 변수형(Variable)  

**상수형(Constant)**  
상수형은 변하지 않는 수를 담아두는 공간이다.  
상수형은 아래와 같이 정의되어있다.  
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)  
 - value: 상수의 값.
 - dtype: 상수의 데이터형. tf.float32 와 같이 실수, 정수등의 데이터 타입을 정의한다.
 - shape: 행렬의 차원을 정의한다. shape=[3,3]으로 정의하면 이 상수는 3 x 3행렬을 저장하게 된다.
 - name: 상수의 이름을 정의한다.

아래 간단한 예제를 확인하여 보자  
```python
a = tf.constant([5])
b = tf.constant([10])
c = tf.constant([2])

d = a * b +c
print(d)
#Tensor("add:0", shape=(1,), dtype=int32)
```
위와같이 실행하게 되면 **Tensor("add:0", shape=(1,), dtype=int32)**의 결과를 얻을 수 있다.  
<span style ="color: red">**수식을 그래프로서만 표현하였지 실질적은 값을 대입하여 계산하지 않았기 때문이다.**</span>  
이러한 실질적인 값을 대입하여 실행하기 위해서는 위에서도 언급하였듯이 Session을 생성하여 실행하면 익숙한 값이 나오게 된다.  
```python
sess = tf.Session()
result = sess.run(d)
print(result)
#52
```
Session으로서 그래프를 실제로 계산을 실행하게 되면 위와 같이 52의 원하는 값을 확인할 수 있다.  

**변수형(Variable)**  
학습을 통해서 구해야 하는 값을 Variable이라고 한다.  
Weight와 Bias가 해당하게 된다.  
<span style ="color: red">**Variable은 선언하게 되면 Variable형의 객체로 생성이 된다.**</span>  

tf.Variable.__init__(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)  

아래 간단한 예제를 확인하여 보자  
```python
var1 = tf.Variable([5])
var2 = tf.Variable([10])
var3 = tf.Variable([2])

var4 = var1 * var2 + var3
print(var4)
#Tensor("add:0", shape=(1,), dtype=int32)
```
Constant와 마찬가지로 그래프만 그리고 실질적인 값을 넣지 않은 상태이므로 다음과 같이 출력되게 된다.  
Session을 통해 실행하게 되면  

```python
sess = tf.Session()
result = sess.run(var4)
print(result)
#Error
```
Session을 통하여 실행하게 되어도 다음과 같은 Error가 계속하여 발생하게 된다.  
```code
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
C:\Tensor\envs\tensorflow\lib\site-packages\tensorflow\python\client\session.py in _do_call(self, fn, *args)
   1333     try:
-> 1334       return fn(*args)
   1335     except errors.OpError as e:

....


```
<span style ="color: red">**Variable은 반드시 초기화가 필요한 Tensor이다. 값이 들어갔다고 해서 초기화됬다는 의미가 아니다.**</span>  
Variable을 초기화하는 명령어는 tf.gloabl_variables_initializer()를 통하여 이루어지게 된다.  
아래 예시를 통해 살펴보자.  
```python
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(var4)
print(result)
#[52]
```
**Variable을 초기화한 뒤 실행**하게 되면 위와 같이 원하는 값을 얻을 수 있게 된다.  

**플레이스 홀더(Placeholder)**  
Input Data를 담아두는 공간이다.  
Placeholder는 Tensor와 Data를 mapping하는 역활을 한다.  

tf.placeholder(dtype=None, shape=None, name='Const', verify_shape=False)   
<span style ="color: red">**위의 선언에서도 살펴볼 수 있듯이 placeholder는 실질적인 값을 정의하지 않는다.**</span>  
실질적인 값을 Mapping을 통해서 이루워지고 이러한 mapping은 **feeding**을 통하여 이루워 진다.  

아래 간단한 예제를 확인하여 보자  
```python
value1 = 5
value2 = 3
value3 = 2

ph1 = tf.placeholder(dtype=tf.float32)
ph2 = tf.placeholder(dtype=tf.float32)
ph3 = tf.placeholder(dtype=tf.float32)

#Ploaceholder는 Tensor와 Data를 mapping하는 역활을 한다.
#feeding이라는 것을 하여야 한다.

feed_dict = {ph1: value1, ph2: value2, ph3: value3}

result_value = ph1 * ph2 * ph3

sess = tf.Session()
result = sess.run(result_value,feed_dict=feed_dict)
print(result)
#30.0
```
feed_dict = {ph1: value1, ph2: value2, ph3: value3}의 명령어를 통해 placeholer와 실질적인 값을 mapping을 실시하게 되고, session에서 **sess.run(result_value,feed_dict=feed_dict)**를 통하여 placeholer의 값을 mapping하는 작업을 하게 되었다.  

<br>
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">

	<tr>	
		<td>상수형(Contant)</td>
		<td>변하지 않는 값 저장</td>
		<td> </td>
	</tr>
	
	<tr>
		<td>플레이서 홀더(Placeholder)</td>
		<td>Input Data(Tensor와 Data를 mapping하는 역활)</td>
		<td>feeding을 통하여 Mapping 필요</td>
	</tr>
	
	<tr>
		<td>변수형(Variable)</td>
		<td>Weight, Bias(학습을 통하여 구해야 하는 값)</td>
		<td>초기화 작업 필요, 객체로서 정의됨</td>
	</tr>
	
	</table>
<br>

### TensorBoard
텐서플로 라이브러리는 학습 결과를 시각화해서 보여주는 강력한 도구인 텐서보드(Tensorboard)를 제공한다.  
이를 통하여 학습이 **올바른 방향으로 진행되고 있는지를 측정**하거나 **실험결과를 분석**하는데 많은 도움을 준다.  
아래 예시는 Tensorboard에서 가장많이 활용되는 예시 2개이다.  
**TensorBoard의 활용 예시**  
1. tf.summary.scalar API를 이용한 step 마다 손실함수 출력
2. 계산 그래프 시각화

위와 같은 **TensorBoard**를 활용하기 위한 과정은 크게 3가지로 나눌 수 있다.  
**TensorBoard 과정**  
1. 무엇을 보고 싶은가?
2. 어디에 기록하고 싶은가?
3. 언제마다 기록하고 싶은가?


**1. 무엇을 보고 싶은가**  
- Scalar값을 보기 위한 Code  
<code>tf.summary.scalar(name,scalar)</code>  
- Image값을 보기 위한 Code  
<code>tf.summary.image(name,image)</code>  
- Histogram을 보기 위한 Code  
<code>tf.summary.histogram(name,histogram)</code>  

**2. 어디에 기록하고 싶은가**  
- 모든 summary값을 통합  
<code>tf.summary.merge_all()</code>  
- 원하는 summary값을 통합  
<code>tf.summary.merge(summaries)</code>  
- 어느폴더에서 기록할 것인가  
<code>tf.summary.FileWriter(log_dir,graph)</code>  

**3. 언제마다 기록하고 싶은가**  
- merge를 실행  
<code>summary = sess.run(merge)</code>  
- summary값을 직접 입력  
<code>tensorboard_writer.add_summary(summary,global_step)</code>  


<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow/blob/master/Basic.ipynb">원본코드</a><br>
참조: <a href="https://bcho.tistory.com/1150">bcho 블로그</a><br>
참조: <a href="https://medium.com/trackin-datalabs/tensorboard-%EA%B0%84%EB%8B%A8%ED%9E%88-%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-18a4fda2efb1">medium 블로그</a><br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
---
layout: post
title:  "TensorFlow-Linear Regression"
date:   2019-08-17 10:00:00 +0700
categories: [Tensorflow]
---

### Linear Regression
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

<span style ="color: red">'선형적이다' </span>라는 표현은 영어로 linear 하다 라고 말한다. linear란 line(선)의 형용사 형태입니다. 이 말에서 유추할 수 있듯이, 선형적이란 것은 어떤 성질이 변하는데 그 변수가 1차원적이다, 즉 어떤 신호에 기울기만 곱한 형태와 같다.  
<div><img src="http://www.rfdh.com/bas_rf/begin/images/linear1.gif" height="200" width="600" /></div>
그림출처:<a href="http://www.rfdh.com/bas_rf/begin/linear.htm">www.rfdh.com </a><br>
간단한 수식으로는 y= bx + a로서 표현할 수 있다.  
여기서 우리가 중점적으로 봐야할 것은 <span style ="color: red">**weight: b, bias: a**</span>의 상수 2개이다.  
이 상수 2개를 찾아낼 수 있으면 우리는 앞으로 Input이 들어올 경우 Output을 구할 수 있다.  
<a href="https://wjddyd66.github.io/r/2019/06/17/Regression.html">회귀분석 자세한 내용</a>
### Linear Regression 예시
아래 표와 같은 DataSet이 있다고 가정하자.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>집넓이</td><td>집값(만원)</td>
	</tr>
	<tr>
		<td>10</td><td>3000</td>
	</tr>
		<tr>
		<td>15</td><td>4000</td>
	</tr>
		<tr>
		<td>16</td><td>12000</td>
	</tr>
			<tr>
		<td>...</td><td>...</td>
	</tr>

	<tr>
		<td>60</td><td>30000</td>
	</tr>
	</tbody>
</table>
<br>
위의 집 넓이를 X, 집값을 Y라 하였을때 다음과 같은 식을 얻을 수 있다.  

$$y= w_1 X + b$$  
위의 식은 아래와 같은 그래프로서 표현할 수 있다.  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/1.png" height="200" width="600" /></div>
실제값과 예측값의 차이는 파란색선의 길이의 합이다.  
실제값과 예측값의 차이는 Cost라고 불리게 되고 이러한 Cost에 대한 식은 아래 식으로 나타낼 수 있다.  

$$f_c(x)=\sum_{i=0}^n  (y_i-\hat{y_i})^2$$  
Loss Function은 아래와 같은 그림으로 나타낼 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/2.PNG" height="250" width="600" /></div>
Cost가 0이 될 확률을 매우 낮지만 <span style ="color: red">**0에 가까울 때(미분했을때의 기울기가 0인 점)**</span>의 값을 구하는 것이 Cost를 최소로 할 수 있다.  
간단하게 구할 수 있다고 생각하지만 Weight가 많아지게 되면 수식이 복잡하게 되므로 Gradient Decent를 사용하게 된다.   
<a href="https://wjddyd66.github.io/dl/2019/07/26/NeuralNetwork-(3)-Optimazation.html">GradientDescent의 자세한 내용</a><br>

### Linear Regression 실제 구현
Linear Regression을 실제로 구현하기 전에 구현하기 위하여 사용된 기법을 먼저 정리하였다.  
**Linear Regression 사용 기법**  
<table class="table">
	<tbody>
	<tr>
		<td>Model</td><td>y= 2 x</td>
	</tr>

	<tr>
		<td>Loss Function</td><td>MSE</td>
	</tr>
	<tr>
		<td>Optimazation</td><td>GradientDescent</td>
	</tr>
	</tbody>
</table>
<br>
**Linear Regression 실제 구현**  
변수(Variable)와 플레이스홀더(placeholder) 를 이용하여
선형회귀 모델의 그래프 구조 (Wx + b) 선언
- W: 가중치
- b: 바이어스
- x: 입력값
- y: 출력값


```python
W = tf.Variable(tf.random_normal(shape=[1]))
b = tf.Variable(tf.random_normal(shape=[1]))
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)
```
Loss Function: MSE 선언  
텐서보드를 위한 요약정보(scalar)선언
```python
loss = tf.reduce_mean(tf.square(linear_model-y))
#텐서 보드를 위한 요약정보 scalar
tf.summary.scalar('loss',loss)
```
텐서 보드를 위한 Merge 및 File경로 설정
```python
merged = tf.summary.merge_all()
tensorboard_writer = tf.summary.FileWriter('./tensorboard_log',sess.graph)
```
Optimier 선언: GradientDescent
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)
```
실제 Data
```python
X_train = [1, 2, 3, 4]
Y_train = [2, 4, 6, 8]
```
Session 실행  
Step마다 merged 연산을 실행하고 저장  
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

X_train = [1, 2, 3, 4]
Y_train = [2, 4, 6, 8]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step,feed_dict={x:X_train, y:Y_train})
    
    #매 Step마다 요약정보 값들을 저장하여 지정된 경로에 저장
    summary = sess.run(merged,feed_dict={x:X_train, y:Y_train})
    tensorboard_writer.add_summary(summary,i)
```
Model Test 및 Session 닫기
```python
X_test = [3.5, 5, 5.5, 6]
print(sess.run(linear_model,feed_dict={x:X_test}))

sess.close()
```
**Linear Regression 결과**  
<table class="table">
	<tbody>
	<tr>
		<td>실제 값</td><td>예측 값</td><td>차이</td>
	</tr>

	<tr>
		<td>7</td><td>7.0019174</td><td>0.0019174</td>
	</tr>
	
	<tr>
		<td>10</td><td>10.007053</td><td>0.007053</td>
	</tr>
	
	<tr>
		<td>11</td><td>11.008765</td><td>0.008765</td>
	</tr>
	
	<tr>
		<td>12</td><td>12.010476</td><td>0.010476</td>
	</tr>
	</tbody>
</table>
<br>

**TensorBoard Loss 결과**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/76.PNG" height="200" width="600" /></div>
<br>


<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow/blob/master/LinearRegression.ipynb">원본코드</a> <br>
참조: <a href="https://www.youtube.com/watch?v=GmtqOlPYB84&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=21">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="https://honeytip91.tistory.com/106">honeytip91 블로그</a> <br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
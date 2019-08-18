---
layout: post
title:  "Tensorflow-ANN"
date:   2019-08-18 11:00:00 +0700
categories: [Tensorflow]
---

### ANN
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

**ANN** 은 딥러닝에서 사용하는 인공신경망(Artificial Neural Network)이다.  
**대표적인 예**와 그에 **해당하는 이론에 대한 내용**은 아래 링크를 참조하자.  
1. <a href="https://wjddyd66.github.io/dl/2019/07/26/Perceptron.html">Perceptron</a>
2. <a href="https://wjddyd66.github.io/dl/2019/07/26/NeuralNetwork-(1)-Basic-&-Activation-Function.html">MLP(NeuralNetwork)</a>

위의 내용에서 이번 Post에서는 **Tensorflow를 활용하여 MLP를 구현**해보자  

### MLP 구현
**MLP** 또한 앞선 Post에서 다룬 내용 **Linear Regression, Logistic Regression**과 같이 **MNIST Data**를 분리하는 예제를 사용한다.  
**MLP** 사용기법은 아래와 같다.  
**MLP 사용기법**
<table class="table">

	<tr>	
		<td>Loss Function</td><td>Softmax-with-Loss</td>
	</tr>
	
	<tr>	
		<td>Optimazation</td><td>Adam</td>
	</tr>
</table>
<br>
**MLP 구현**  
MLP Model
텐서플로 라이브러리를 import
```python
import tensorflow as tf
```
MNIST Data를 다운로드 및 불러오기(One-Hot-Encoding O)
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)
```
학습을 위한 Parameter 선언
- num_epochs: 학습횟수
- batch_size: 배치개수
- display_step: 손실 함수 출력 주기
- input_size: 28 * 28 =784


```python
learning_rate = 0.001
num_epochs = 30
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10
```
입력값과 출력값을 받기 위해 Placeholder 선언

```python
x = tf.placeholder(tf.float32, shape = [None,input_size])
y = tf.placeholder(tf.float32, shape = [None, output_size])
```
ANN Model 정의
- Activation Function: ReLu


```python
def build_ANN(x):
    #Layer1
    W1 = tf.Variable(tf.random_normal(shape = [input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape = [hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x,W1)+b1)
    #Layer 2
    W2 = tf.Variable(tf.random_normal(shape = [hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape = [hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output,W2)+b2)
    #Layer 3
    W_output = tf.Variable(tf.random_normal(shape = [hidden2_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape = [output_size]))
    logits = tf.matmul(H2_output,W_output)+b_output
    
    return logits
```

실제 ANN Model 선언
```python
predicted_value = build_ANN(x)
```
- Loss Function: Softmax with Loss
- Optimazer: Adam


```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
```
세션 생성 및 실행  
Model 성능 Test
```python
with tf.Session() as sess:
    #변수들에 초기값을 할당
    sess.run(tf.global_variables_initializer())
    
    #지정된 횟수만큼 최적화 수행
    for epoch in range(num_epochs):
        average_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #모든 배치들에 대해서 최적화를 수행
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step,loss],feed_dict = {x: batch_x, y: batch_y})
            #평균 손실 측척
            average_loss += current_loss/total_batch
            #지정된 epoch 마다 학습결과 출력
        if epoch % display_step == 0:
            print('반복(Epoch): {}, 손실함수(Loss): {}'.format(epoch+1, average_loss))
    #Model 성능 Test
    correct_prediction = tf.equal(tf.arg_max(predicted_value,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('정확도(Accuract): %f'%(accuracy.eval(feed_dict={x:mnist.test.images, y: mnist.test.labels})))
```
**결과**  
```code
반복(Epoch): 1, 손실함수(Loss): 289.31592089216275
반복(Epoch): 2, 손실함수(Loss): 66.723739820106
반복(Epoch): 3, 손실함수(Loss): 43.624182897193386
반복(Epoch): 4, 손실함수(Loss): 32.35598173319737
반복(Epoch): 5, 손실함수(Loss): 25.1575914409673

...

반복(Epoch): 26, 손실함수(Loss): 0.3990266389692753
반복(Epoch): 27, 손실함수(Loss): 0.3039966435727189
반복(Epoch): 28, 손실함수(Loss): 0.2636221467217358
반복(Epoch): 29, 손실함수(Loss): 0.2895586178715198
반복(Epoch): 30, 손실함수(Loss): 0.2183748629522514
```
**정확도(Accuract): 0.945500**
<br><br>


<hr>
참조: <a href="https://github.com/wjddyd66/Tensorflow/blob/master/ANN.ipynb">원본코드</a> <br>
참조: 텐서플로로 배우는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
---
layout: post
title:  "Tensorflow-RNN"
date:   2019-08-29 12:00:00 +0700
categories: [Tensorflow]
---

### RNN
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
RNN(Recurrent Neural Network)이란 **순차적인 정보**를 처리하는 데 있다.  
즉, 이전 까지 반복하였던 **상관없는 두 변수간의 값으로 인한 정보를 처리하는 것이 아닌 한 정보에 대한 특정 Domain의 값을 나타내는 정보를 처리하는 것** 이다.  
예를 들어, 문장에서 다음에 나올 단어를 추측하고 싶다면 이전에 나온 단어들을 아는 것이 큰 도움이 될 것이다.  
또한 집의 가격이 어떻게 변할지에 대하여 한 집의 가격을 계속 관찰하게 되면 집의 가격이 **시간(Domain)에 따라 어떻게 변하는지 예측**할 수 있을 것이다.  
위의 예시가 가능한 이유는 동일한 Task에 대하여 **하나의 Hidden Layer**를 계속하여 Trainning 하기 떄문이다.  
출력 결과는 이전의 계산 결과에 영향을 받기 때문에 RNN은 현재지 계산된 결과에 대한 "메모리" 정보를 갖고 있다고 볼 수도 있다.  
RNN의 구조는 아래와 같이 나타낸다.  
<div><img  src="http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg" width="100%" height="100%"></div>
- <span>$$x_t$$</span>는 시간 스텝(time step) t 에서의 입력값이다.
- <span>$$x_t$$</span>는 시간 스텝(time step) t 에서의 Hidden state이다. 네트워크의 "메모리" 부분으로서, 이전 시간의 스텝의 hidden state 값과 현재 시간 스텝의 입력값에 의해 계산된다.  
<span>$$s_t = f(Ux_t + Ws_{t-1}) \text{   f는 tanh or ReLU}$$</span>
- <span>$$o_t$$</span>는 시간 스텝(time step)에서의 출력값이다. 예를 들어 다음 단어를 추축하고 싶다면 단어 수만큼의 차원의 확률 벡터가 될 것이다. <span>$$o_t = softmax(Vs_t)$$</span>

몇 가지 짚어두고 넘어갈 점이 있다.  
- Hidden state s_t는 네트워크의 메모리라고 생각할 수 있다. - s_t는 과거의 시간 스텝들에서 일어난 일들에 대한 정보를 전부 담고 있고, 출력값 o_t는 오로지 현재 시간 스텝 t의 메모리에만 의존한다. 하지만 위에서 잠깐 언급했듯이, 실제 구현에서는 너무 먼 과거에 일어난 일들은 잘 기억하지 못한다.  
- 각 layer마다의 파라미터 값들이 전부 다 다른 기존의 deep한 신경망 구조와 달리, RNN은 모든 시간 스텝에 대해 파라미터 값을 전부 공유하고 있다 (위 그림의 U, V, W). 이는 RNN이 각 스텝마다 입력값만 다를 뿐 거의 똑같은 계산을 하고 있다는 것을 보여준다. 이는 학습해야 하는 파라미터 수를 많이 줄여준다.  
- 위 다이어그램에서는 매 시간 스텝마다 출력값을 내지만, 문제에 따라 달라질 수도 있다. 예를 들어, 문장에서 긍정/부정적인 감정을 추측하고 싶다면 굳이 모든 단어 위치에 대해 추측값을 내지 않고 최종 추측값 하나만 내서 판단하는 것이 더 유용할 수도 있다. 마찬가지로, 입력값 역시 매 시간 스텝마다 꼭 다 필요한 것은 아니다. RNN에서의 핵심은 시퀀스 정보에 대해 어떠한 정보를 추출해 주는 hidden state이기 때문이다.  
출처: <a href="http://aikorea.org/blog/rnn-tutorial-1/">aikorea</a><br>



### RNN BackPropagation  
RNN을 아래와 같은 그림으로 같단히 나타내어 보자.  
<div><img src="http://www.wildml.com/wp-content/uploads/2015/10/rnn-bptt1.png" height="250" width="600" /></div><br>

위와 같은 그림에서 다음과 같은 식을 정의하고 가자  
Activation Function: tanh  
Classifier: Softmax  
<p>$$s_t = tanh(Ux_t + Ws_{t-1})$$</p>
<p>$$\hat{y_t} = softmax(Vs_t) $$</p>
<p>$$y_t \text{: 시간 스텝 t 에서 실제 단어, } \hat{y_t} \text{: 예측값}$$</p>
Loss Function: Cross Entropy  
<p>$$E(y_t,\hat{y_t}) = -y_t log(\hat{y_t})$$</p>
<p>$$E(y,\hat{y}) = -\sum_t{E(y_t,\hat{y_t})}$$</p>
<p>$$= -\sum_t{-y_t log(\hat{y_t})}$$</p>

**Parameter U, V, W 에 대한 Error 의 Gradient 를 계산하고 SGD를 이용하여 Parameter를 최적화 하여 Loss를 적게 만드는 것이 목표이다.**  

**1. Parameter V**  
<p>$$\frac{\partial E_3}{\partial V} = \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial V}$$</p>
<p>$$= \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial z_3} \frac{\partial z_3}{\partial V} $$</p>
<p>$$= (\hat{y_3} - y_3) \bigotimes s_3$$</p>
<p>$$ z_3 = Vs_3$$</p>
위의 식에서 <span>$$\frac{\partial E_3}{\partial \hat{y+3}} \frac{\partial \hat{y_3}}{\partial z_3}$$</span>의 경우 Softmax-with-Loss의 역전파로서 계산 과정을 건너 뛰었다.  
<a href="https://wjddyd66.github.io/tensorflow/2019/08/18/Logistic-Regression.html">Softmax-with-Loss의 역전파</a><br>

위의 식에서 주목해야 할 점은 **<span>$$\frac{\partial E_3}{\partial V}$$</span>은 현재 시간 스탭의 <span>$$\hat{y_3}, y_3, s_3$$</span>**에만 의존한다는 것이다.  
즉 **V Parameter를 갱신하는 것은 현재 시간 스탭의 값만 알아도 수행할 수 있다는 점** 이다.  

**2. Parameter W, U**  
**W, U**에 대해서 정리하면 **V**처럼 **현재 시간 스탭의 값만 알아도 수행할 수 없다는 것**을 알 수 있다.  
아래의 식으로서 살펴보자  
<p>$$\frac{\partial E_3}{\partial W} = \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial s_3} \frac{\partial s_3}{\partial W}$$</p>
여기서 <span>$$s_t = tanh(Ux_t + Ws_{t-1})$$</span>이므로
<span>$$s_3$$</span>는 <span>$$s_2$$</span>에 의존하고 <span>$$s_2$$</span>는 <span>$$s_1$$</span>에 의존하는 현상이 발생하게 된다.  
이러한 상황으로 인하여 **Chain Rule**이 계속해서 이어지는 것을 알 수 있다.  

아래 식을 살펴보게 되면 **Chain Rule**을 적용한 식을 알 수 있다.  
<p>$$\frac{\partial E_3}{\partial W} = \sum_{k=0}^{3} \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial s_3} \frac{\partial s_3}{\partial s_k} \frac{\partial s_k}{\partial W} $$</p>
위의 식을 살펴보게 되면 **각 시간 스텝이 gradient에 기여**하는 것을 전부 더해준다.  
즉, W는 우리가 현재 처리중인 출력 부분까지의 모든 시간 스템에서 사용되기 때문에, t=3 부터 t=0 까지 gradient들을 전부 backpropagat해 주어야 한다.  
<div><img src="http://www.wildml.com/wp-content/uploads/2015/10/rnn-bptt-with-gradients.png" height="250" width="600" /></div><br>
위를 살펴보게 되면 기존 Neural Network에서 적용되는 backpropagate의 과정과 같은 것을 알 수 있다.  
<p>$$z_t = Ux_t + WS_{t-1} \text{이라고 치환}$$</p>
<p>$$\delta_3^3 = \frac{\partial E_3}{\partial z_3}$$</p>
<p>$$ = \frac{\partial E_3}{\partial \hat{y_3}} \frac{\partial \hat{y_3}}{\partial z_3}$$</p>
<p>$$\text{softmax_with_crossentropy backpropagation을 적용하면}$$</p>
<p>$$ = (\hat{y_3} - y_3)s_3$$</p>  

<p>$$\delta_2^3 = \frac{\partial E_3}{\partial z_2}$$</p>
<p>$$ = \frac{\partial E_3}{\partial z_3} \frac{\partial z_3}{\partial s_2} \frac{\partial s_2}{\partial z_2}$$</p>  
<p>$$= \delta_3^3 \frac{\partial z_3}{\partial s_2} \frac{\partial s_2}{\partial z_2}$$</p>  

<p>$$\delta_1^3 = \frac{\partial E_3}{\partial z_1}$$</p>
<p>$$ = \frac{\partial E_3}{\partial z_2} \frac{\partial z_2}{\partial s_1} \frac{\partial s_1}{\partial z_1}$$</p>  
<p>$$= \delta_2^3 \frac{\partial z_2}{\partial s_1} \frac{\partial s_1}{\partial z_1}$$</p>  

위의 식을 유도하였으면 아래와 같은 식을 최종적으로 얻어 낼 수 있다.  
i 는 특정 시간 스탭이라고 하면  
<p>$$\frac{\partial E_3}{\partial U} = \delta_i^3 x_i^{T}$$</p>
<p>$$\frac{\partial E_3}{\partial W} = \delta_i^3 s_{i-1}^{T}$$</p>
RNN이므로 계산된 값을 모두 더해주는 것을 말고는 Neural Network의 Backpropagation과 같은 식이 유도되는 것을 알 수 있다.  

### Long Term Dependency
RNN의 장점 중 하나는 이전 정보를 현재 작업으로 연결할 수 있다는 점 이다.(Memory 사용)  
하지만 이러한 장점으로 인한 단점이 생기게 되는 것이 **Long Term Dependency**이다.  
첫번째의 예를 생각해보자.  
우리가 현재 시점의 뭔가를 얻기 위해서 멀지 않은 최근의 정보만 필요로 할 때도 있다. 예를 들어 이전 단어들을 토대로 다음에 올 단어를 예측하는 언어 모델을 생각해 보자. 만약 우리가 "the clouds are in the sky"에서의 마지막 단어를 맞추고 싶다면, 저 문장 말고는 더 볼 필요도 없다. 마지막 단어는 sky일 것이 분명하다. 이 경우처럼 필요한 정보를 얻기 위한 시간 격차가 크지 않다면, RNN도 지난 정보를 바탕으로 학습할 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/79.PNG" height="250" width="600" /></div>
<br>
두번째의 예를 생각해보자.  
하지만 반대로 더 많은 문맥을 필요로 하는 경우도 있다. "I grew up in France... I speak fluent French"라는 문단의 마지막 단어를 맞추고 싶다고 생각해보자. 최근 몇몇 단어를 봤을 때 아마도 언어에 대한 단어가 와야 될 것이라 생각할 수는 있지만, 어떤 나라 언어인지 알기 위해서는 프랑스에 대한 문맥을 훨씬 뒤에서 찾아봐야 한다. 이렇게 되면 필요한 정보를 얻기 위한 시간 격차는 굉장히 커지게 된다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/80.PNG" height="250" width="600" /></div>
<br>
이러한 장기 의존성의 문제를 해결하기 위해 나온 RNN을 활용한 Model이 **LSTM**이다.
<br><br>  

### LSTM
RNN이 많이 사용되는 자연어 처리에서의 예를 들어보자.  
먼저 텍스트 데이터는 이미지와는 다르게 서로 인접한 단어와 Discrete 하다는 특성을 갖는다. 이는 특정단어또는 문장이 주변부 단어 또는 문장과 연관성이 높을 수도 있고 낮을 수도 있음을 의미하며 생성형 요약을 구현하기 위해서는 문장의 길이가 길어져도 각각의 정보를 오래 기억할 수 있어야한다.  
예를 들어“나는 학교에서 밥을 영희와 먹었다.”라는 문장에서 주어인 ‘나’와 ‘영희’는 위치적으로서로가장 떨어져 먼 거리에 위치해있다.  
그런데 주어인 나와 영희의 관계가 가족이라면 나머지 단어들 ‘학교’, ‘밥’ 등 보다 ‘나’는 ‘영희’와 연관성이 더 높다.  
신경망은 이 문장을 받아들일때 ‘나’ 라는 정보를 멀리 떨어진 ‘영희’ 라는 정보와 연관 지을 수있어야한다.  
이러한 텍스트가 갖는 특징에대한 방안으로 Hidden Layer에 LSTM 을 사용하게 된다.  
이러한 LSTM은 아래와 같은 그림으로서 표현할 수 있다.  
<div><img src="http://i.imgur.com/jKodJ1u.png" height="100%" width="100%" /></div>

**LSTM 은 장기 의존성을 해결하기 위한 방안이다. LSTM 은 Forget gate와 Input gate가 특징으로 Forget gate는 과거 정보를 잊기 위한 게이트이다. Input gate 는 현재 정보를 기억하기 위한 게이트이다.**  
두 게이트 모두 앞에 σ(시그모이드)를곱하여 0~1사이에값을 가지게 된다.  
σ(시그모이드)를 곱한 Forget gate 와Input gate를 통해 과거의 정보와 현재의 정보를 얼마나 기억할 것인가를 정하여 장기 의존성의 문제를 해결한다.  
이러한 Forget gate와 Input gate는 아래와 같은 그림으로서 나타낼 수 있다.  
<div><img src="http://i.imgur.com/MPb3OvZ.png" height="100%" width="100%" /></div>
- Forget gate: <span>$$f_t = \sigma (W_{xh_f}x_t +W_{hh_f}h_{t-1} + b_{h_f})$$</span>
- Input gate: <span>$$i_t = \sigma (W_{xh_i}x_t +W_{hh_i}h_{t-1} + b_{h_i})$$</span>
<br><br>  

### 임베딩(Embedding)
**Embedding**이란 Machine Learning Algorithm에서 자연어 처리 문제를 다룰 떄 널리 사용되는 기법이다. 
이전 까지 사용하였던 방법은 One-hot-Encoding 이다.  
**One-hot-Encoding**이란 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다.  
이러한 방식은 크게 2가지의 단점이 생기게 된다.  
1. 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다.
2. 단어의 유사성을 전혀 표현하지 못한다.

이러한 One-hot-Encoding은 하나의 원소만 1이고 나머지의 원소는 0이므로 Sparse하다고 표현된다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/81.PNG" height="250" width="600" /></div>

이러한 Sparse한 표현 형태를 Dense한 **임베딩 행렬**을 곱해서 아래와 같이 나타낼 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/82.PNG" height="250" width="600" /></div>
이러한 Dense한 Embedding은 다음과 같은 3가지의 장점을 가지게 된다.  
1. 데이터의 표현 형태를 Sparse한 형태에서 Dense한 형태로 바꿔서더욱 효율적인 학습이 가능하도록 만들어 준다.
2. 데이터의 차원을 축소해서 연산량을 감소시킨다.
3. 단어 사이의 유사성을 표현할 수 있다.

위와 같은 Embedding의 과정은 아래의 Tensorflow API로서 구현될 수 있다.  
**Embedding Tensorflow API**  
```python
tf.nn.embedding_lookup(params, ids, name=None)
```
- params: 임베딩을 적용할 임베딩 핼영 텐서
- ids: 스칼라형태의 One-hot-Encoding으로 표현된 임베딩을 적용할 인풋 데이터
- name: 연산의 이름 (optional)

**Embedding 구현**  
텐서플로와 numpy 라이브러리를 임포트
```python
import tensorflow as tf
import numpy as np
```

원본 데이터의 전체 단어 개수(One-hot-Encoding 표현의 차원)와 축소 할 임베딩 차원을 정의
```python
vocab_size = 100 # One-hot-encoding 된 vocab 크기
embedding_size = 25 # 임베딩된 vocab 크기
```

스칼라 형태의 One-hot-Encoding 인풋 데이터를 받기 위한 플레이스홀더를 정의
```python
inputs = tf.placeholder(tf.int32, shape=[None])
```

임베딩 행렬은 선언하고 tf.nn.embedding_lookup API를 이용해서 임베딩을 수행
```python
embedding = tf.Variable(tf.random_normal([vocab_size, embedding_size]), dtype=tf.float32)
embedded_inputs = tf.nn.embedding_lookup(embedding, inputs)
```

세션을 열고 변수들을 초기화
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

샘플 인풋 데이터를 정의하고, 임베딩을 수행해서 변형 결과를 확인
```python
input_data = np.array([7])
print("Embedding 전 인풋 데이터: ")
print(sess.run(tf.one_hot(input_data,vocab_size)))
print(tf.one_hot(input_data,vocab_size).shape)
print('Embedding 결과: ')
print(sess.run([embedded_inputs],feed_dict={inputs:input_data}))
print(sess.run([embedded_inputs],feed_dict={inputs:input_data})[0].shape)

input_data = np.array([7,11,67,42,21])
print("Embedding 전 인풋 데이터: ")
print(sess.run(tf.one_hot(input_data,vocab_size)))
print(tf.one_hot(input_data,vocab_size).shape)
print('Embedding 결과: ')
print(sess.run([embedded_inputs],feed_dict={inputs:input_data}))
print(sess.run([embedded_inputs],feed_dict={inputs:input_data})[0].shape)
```

```code
Embedding 전 인풋 데이터: 
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0.]]
(1, 100)
Embedding 결과: 
[array([[-0.51674426,  0.00487127,  0.84068084,  0.9302916 , -1.8013382 ,
         0.36010912, -0.6714361 , -1.1757022 ,  0.5983085 , -1.510787  ,
        -1.3008538 ,  0.7145155 ,  0.11995181,  1.5791241 ,  1.8553414 ,
        -0.27274016,  0.06851366,  0.28717   ,  0.8271994 ,  0.2263984 ,
        -0.24412726,  0.4041885 , -1.751218  , -0.35375005, -0.5199548 ]],
      dtype=float32)]
(1, 25)

Embedding 전 인풋 데이터: 
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.

...

 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0.]]
(5, 100)
Embedding 결과: 
[array([[-0.51674426,  0.00487127,  0.84068084,  0.9302916 , -1.8013382 ,
         0.36010912, -0.6714361 , -1.1757022 ,  0.5983085 , -1.510787  ,
        -1.3008538 ,  0.7145155 ,  0.11995181,  1.5791241 ,  1.8553414 ,
        -0.27274016,  0.06851366,  0.28717   ,  0.8271994 ,  0.2263984 ,

...

         1.6010855 , -0.19003063, -1.2525078 ,  0.39467615, -1.9226099 ,
         0.1577744 ,  0.62687063,  0.7255562 ,  0.49477506,  1.6029706 ]],
      dtype=float32)]
(5, 25)
```
<br><br>

### RNN 구현(Char-RNN)
Char-RNN은 하나의 글자를 RNN의 입력값으로 받고, RNN은 다음에 올 글자를 예측하는 문제이다.  
이를 위해서 RNN의 타겟 데이터를 인풋 문장에서 한 글자씩 뒤로 민 형태로 구성하면 된다.  
Char-RNN의 출력값 형태는 학습에 사용하는 전체문자 집합에 대한 SOftmax 출력값이 된다.  

<div><img src="http://tommymullaney.com/img/google-hangouts-feature.png" height="250" width="600" /></div>

<br>
먼저 Data 전처리를 위한 과정인 **utils.py**를 살펴보자  

#### utils.py  
필요한 라이브러리를 임포트
```python
import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
```

TextLoader 클래스를선언  
__init__: 생성자  
- data_dir: 데이터 경로
- batch_size: 배치 크기
- seq_length: 시계열 데이터 길이 = Input Data로 받아들일 수 있는 길이라고 생각하면 된다.


기존에 전처리가 진행된 파일(vocab.pkl, data.npy)이 존재하면 불러오고 없으면 전처리 수행하여 vocab.pkl, data.npy 파일 생성  

```python
class TextLoader():
  def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.encoding = encoding

    input_file = os.path.join(data_dir, "input.txt")
    vocab_file = os.path.join(data_dir, "vocab.pkl")
    tensor_file = os.path.join(data_dir, "data.npy")

    # 전처리된 파일들("vocab.pkl", "data.npy")이 이미 존재하면 이를 불러오고 없으면 데이터 전처리를 진행합니다.
    if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
      print("reading text file")
      self.preprocess(input_file, vocab_file, tensor_file)
    else:
      print("loading preprocessed files")
      self.load_preprocessed(vocab_file, tensor_file)
      # 배치를 생성하고 배치 포인터를 배치의 시작지점으로 리셋합니다.
      self.create_batches()
      self.reset_batch_pointer()
```

데이터 전처리를 진행하는 **preprocess** method 정의  
input.txt 파일을 읽어 전체 문제들에 대하여  vocab Dictionary를 만들고, 텍스트 데이터를 id 형태로 바꿔서 data.npy파일로 저장한다.  
- collection.Count() => 각각의 요소가 몇개 인지 파악
- .items() => Key, Value를 쌍으로 가져온다.
- zip()=> 동일한 개수로 이루어진 자료형을 묶어 주는 역할

여기서의 id는 단어별 반복 횟수라고 생각할 수 있다.
```python
  # 데이터 전처리를 진행합니다.
  def preprocess(self, input_file, vocab_file, tensor_file):
    with codecs.open(input_file, "r", encoding=self.encoding) as f:
      data = f.read()
      # 데이터에서 문자(character)별 등장횟수를 셉니다.
      counter = collections.Counter(data)
      count_pairs = sorted(counter.items(), key=lambda x: -x[1])
      self.chars, _ = zip(*count_pairs) # 전체 문자들(Chracters)
      self.vocab_size = len(self.chars) # 전체 문자(단어) 개수
      self.vocab = dict(zip(self.chars, range(len(self.chars)))) # 단어들을 (charcter, id) 형태의 dictionary로 만듭니다.
      # vocab dictionary를 "vocab.pkl" 파일로 저장합니다.
      with open(vocab_file, 'wb') as f:
        cPickle.dump(self.chars, f)
      # 데이터의 각각의 character들을 id로 변경합니다.
      self.tensor = np.array(list(map(self.vocab.get, data)))
      # id로 변경한 데이터를 "data.npy" binary numpy 파일로 저장합니다.
      np.save(tensor_file, self.tensor)
```
**Vocab File**  
```code
[(' ',
  'e',
  't',
  'o',
  'a',

...

  'Q',
  'Z',
  'X',
  '3',
  '&',
  '$')]
```
**data File**  
```code
array([50,  9,  7, ..., 26, 10, 11])
```
미리 전처리된 vocab.pkl 파일이 존재하면 해당 파일들을 읽어서 변수에 할당
```python
  # 전처리한 데이터가 파일로 저장되어 있다면 파일로부터 전처리된 정보들을 읽어옵니다.
  def load_preprocessed(self, vocab_file, tensor_file):
    with open(vocab_file, 'rb') as f:
      self.chars = cPickle.load(f)
      self.vocab_size = len(self.chars)
      self.vocab = dict(zip(self.chars, range(len(self.chars))))
      self.tensor = np.load(tensor_file)
      self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
```

데이터를 배치 단위로 묶고, 인풋 데이터를 한 글자씩 뒤로 민 형태로 타겟 데이터를 구성
```python
  # 전체 데이터를 배치 단위로 묶습니다.
  def create_batches(self):
    self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))

    # 데이터 양이 너무 적어서 1개의 배치도 만들수없을 경우, 에러 메세지를 출력합니다.
    if self.num_batches == 0:
      assert False, "Not enough data. Make seq_length and batch_size small."

    # 배치에 필요한 정수만큼의 데이터만을 불러옵니다. e.g. 1115394 -> 1115000
    self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    xdata = self.tensor
    ydata = np.copy(self.tensor)
    # 타겟 데이터는 인풋 데이터를 한칸 뒤로 민 형태로 구성합니다.
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    # batch_size 크기의 배치를 num_batches 개수 만큼 생성합니다. 
    self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
      self.num_batches, 1)
    self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
      self.num_batches, 1)
```

다음 배치를 불러오는 next_batch 함수와 배치의 인덱스를 1번째 배치로 리셋하는 reset_batch_pointer 함수를 정의
```python
  # 다음 배치롤 불러오고 배치 포인터를 1만큼 증가시킵니다.  
  def next_batch(self):
    x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
    self.pointer += 1
    return x, y

  # 배치의 시작점을 데이터의 시작지점으로 리셋합니다.
  def reset_batch_pointer(self):
    self.pointer = 0
```

#### Char-RNN  
유틸리티 함수를 모아놓은 utils 모듈에서 학습 데이터를 읽고 전처리 하기 위한 TextLoader 클래스를 임포트
```python
from utils import TextLoader
```

학습에 필요한 설정값들을 지정  
RNN의 경우 시계열 데이터를 다루기 때문에 시계열 길이를 나타내는 차원 seq_length 추가

```python
# 학습에 필요한 설정값들을 지정합니다.
data_dir = 'data/tinyshakespeare'
#data_dir = 'data/linux'
batch_size = 50 # Training : 50, Sampling : 1
seq_length = 50 # Training : 50, Sampling : 1
hidden_size = 128   # 히든 레이어의 노드 개수
learning_rate = 0.002
num_epochs = 2
num_hidden_layers = 2
grad_clip = 5   # Gradient Clipping에 사용할 임계값
```

TextLoader 클래스를 이용해서 학습 데이터를 불러온다.  
학습 데이터에 포함된 모든 단어들을 포함한 List인 Chars 와 id를 부여해 Dictionary 형태로 만든 vocab을 선언
```python
# TextLoader를 이용해서 데이터를 불러옵니다.
data_loader = TextLoader(data_dir, batch_size, seq_length)
# 학습데이터에 포함된 모든 단어들을 나타내는 변수인 chars와 chars에 id를 부여해 dict 형태로 만든 vocab을 선언합니다.
chars = data_loader.chars 
vocab = data_loader.vocab
vocab_size = data_loader.vocab_size # 전체 단어개수
```
인풋 데이터와 타겟 데이터, 초기 상태값을 입력 받기 위한 플레이스 홀더 선언  
batch_size와 Input Data의 크기인 seq_length로 나중에 치환하기 위하여 None으로서 선언
```python
# 인풋데이터와 타겟데이터, 배치 사이즈를 입력받기 위한 플레이스홀더를 설정합니다.
input_data = tf.placeholder(tf.int32, shape=[None, None])  # input_data : [batch_size, seq_length])
target_data = tf.placeholder(tf.int32, shape=[None, None]) # target_data : [batch_size, seq_length])
state_batch_size = tf.placeholder(tf.int32, shape=[])      # Training : 50, Sampling : 1
```

Hidden_Layer의 크기는 RNN 마지막 은닉층 출력값을 vocab_size만큼의 소프트맥스 행렬로 변환하기 위한 변수들 설정
```python
# RNN의 마지막 히든레이어의 출력을 소프트맥스 출력값으로 변환해주기 위한 변수들을 선언합니다.
# hidden_size -> vocab_size
softmax_w = tf.Variable(tf.random_normal(shape=[hidden_size, vocab_size]), dtype=tf.float32)
softmax_b = tf.Variable(tf.random_normal(shape=[vocab_size]), dtype=tf.float32)
```

vocab_size 크기의 인풋 데이터를 hidden_size 크기로 임베딩을 수행  
인풋 데이터의 임베딩된 형태인 inputs 변수를 RNN의 입력값으로 사용
```python
# 인풋데이터를 변환하기 위한 Embedding Matrix를 선언합니다.
# vocab_size(One-Hot Encoding 차원) -> hidden_size (Embedded 차원)
embedding = tf.Variable(tf.random_normal(shape=[vocab_size, hidden_size]), dtype=tf.float32)
inputs = tf.nn.embedding_lookup(embedding, input_data)
```

즉 현재 RNN의 구조를 보게 되면  
Input(batch_size, seq_length) => One-hot-Encoding(vocab size) => Embedding(hidden_size)  
=> Hidden Layer(Hidden Size, Num_Hidden_Size(2))  
=> Softmax Layer(Hidden Size -> Vocab Size)  
가 반복된다는 것을 알 수 있다.  
Num_Hidden_Size(2)는 CNN처럼 문장에서 Feature의 개수 라고 생각할 수 있다.  
Softmax 에서 차원이 늘어나는 이유는 전체 Voacb 중에서 일부를 사용하기 때문이다.  
```python
# 초기 state 값을 0으로 초기화합니다.
initial_state = cell.zero_state(state_batch_size, tf.float32)

# 학습을 위한 tf.nn.dynamic_rnn을 선언합니다.
# outputs : [batch_size, seq_length, hidden_size]
outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
# ouputs을 [batch_size * seq_length, hidden_size] 형태로 변환합니다.
output = tf.reshape(outputs, [-1, hidden_size])

# 최종 출력값을 설정합니다.
# logits : [batch_size * seq_length, vocab_size]
logits = tf.matmul(output, softmax_w) + softmax_b
probs = tf.nn.softmax(logits)
```
- Loss: Softmax_with_loss
- Optimizer: AdamOptimizer


```python
# Cross Entropy 손실 함수를 정의합니다. 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_data))

# 옵티마이저를 선언하고 옵티마이저에 Gradient Clipping을 적용합니다.
# grad_clip(=5)보다 큰 Gradient를 5로 Clipping합니다.
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.apply_gradients(zip(grads, tvars))
```

세션을 열고 학습을 진행한다.  
여기서는 아까 과정에서 얘기한  
Input(batch_size, seq_length) => One-hot-Encoding(vocab size)
을 통하여 현재 InputData를 Embedding 하기 위한 전처리 과정을 진행하는 것을 알 수 있다.  

학습이 끝나면, 학습된 파라미터를 이용해서 새로운 문장을 생성하기 위한 샘플링을 진행하게 되는 것을 알 수 있다.  
각각의 샘플링 타입의 특징을 살펴보면 argmax를 취해 정확히 다음에 올 글자를 예측할 수 있지만, 특정 알파벳 이후에는 항상 똑같은 특정 알파벳이 뽑혀서 생성되는 문장의 자유도가 떨어지는 것을 확인 할 수 있다.  
```python
# 세션을 열고 학습을 진행합니다.
with tf.Session() as sess:
  # 변수들에 초기값을 할당합니다.
  sess.run(tf.global_variables_initializer())
  
  for e in range(num_epochs):
    data_loader.reset_batch_pointer()
    # 초기 상태값을 초기화합니다.
    state = sess.run(initial_state, feed_dict={state_batch_size : batch_size})

    for b in range(data_loader.num_batches):
      # x, y 데이터를 불러옵니다.
      x, y = data_loader.next_batch()
      # y에 one_hot 인코딩을 적용합니다. 
      y = tf.one_hot(y, vocab_size)            # y : [batch_size, seq_length, vocab_size]
      y = tf.reshape(y, [-1, vocab_size])       # y : [batch_size * seq_length, vocab_size]
      y = y.eval()

      # feed-dict에 사용할 값들과 LSTM 초기 cell state(feed_dict[c])값과 hidden layer 출력값(feed_dict[h])을 지정합니다.
      feed_dict = {input_data : x, target_data: y, state_batch_size : batch_size}
      for i, (c, h) in enumerate(initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      # 파라미터를 한스텝 업데이트합니다.
      _, loss_print, state = sess.run([train_step, loss, final_state], feed_dict=feed_dict)

      print("{}(학습한 배치개수)/{}(학습할 배치개수), 반복(epoch): {}, 손실함수(loss): {:.3f}".format(
        e * data_loader.num_batches + b,
        num_epochs * data_loader.num_batches,
        (e+1), 
        loss_print))

    print("트레이닝이 끝났습니다!")   


  # 샘플링을 시작합니다.
  print("샘플링을 시작합니다!")
  num_sampling = 4000  # 생성할 글자(Character)의 개수를 지정합니다. 
  prime = u' '         # 시작 글자를 ' '(공백)으로 지정합니다.
  sampling_type = 1    # 샘플링 타입을 설정합니다.
  state = sess.run(cell.zero_state(1, tf.float32)) # RNN의 최초 state값을 0으로 초기화합니다.

  # Random Sampling을 위한 weighted_pick 함수를 정의합니다.
  def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

  ret = prime       # 샘플링 결과를 리턴받을 ret 변수에 첫번째 글자를 할당합니다.
  char = prime      # Char-RNN의 첫번째 인풋을 지정합니다.  
  for n in range(num_sampling):
    x = np.zeros((1, 1))
    x[0, 0] = vocab[char]

    # RNN을 한스텝 실행하고 모델이 예측한 Softmax 행렬을 리턴으로 받습니다.
    feed_dict = {input_data: x, state_batch_size : 1, initial_state: state}
    [probs_result, state] = sess.run([probs, final_state], feed_dict=feed_dict)         

    # 불필요한 차원을 제거합니다.
    # probs_result : (1,65) -> p : (65)
    p = np.squeeze(probs_result)

    # 샘플링 타입에 따라 3가지 종류 중 하나로 샘플링 합니다.
    # sampling_type : 0 -> 다음 글자를 예측할때 항상 argmax를 사용
    # sampling_type : 1(defualt) -> 다음 글자를 예측할때 항상 random sampling을 사용
    # sampling_type : 2 -> 다음 글자를 예측할때 이전 글자가 ' '(공백)이면 random sampling, 그렇지 않을 경우 argmax를 사용
    if sampling_type == 0:
      sample = np.argmax(p)
    elif sampling_type == 2:
      if char == ' ':
        sample = weighted_pick(p)
      else:
        sample = np.argmax(p)
    else:
      sample = weighted_pick(p)

    pred = chars[sample]
    ret += pred     # 샘플링 결과에 현재 스텝에서 예측한 글자를 추가합니다. (예를들어 pred=L일 경우, ret = HEL -> HELL)
    char = pred     # 예측한 글자를 다음 RNN의 인풋으로 사용합니다.

    print("샘플링 결과:")
    print(ret)
```
**결과**  
```code
샘플링 결과:
 s
샘플링 결과:
 sl
샘플링 결과:
 sla
샘플링 결과:
 slaw
샘플링 결과:
 slawb
샘플링 결과:
 slawbl
 
 ...
 
 
 slawbles
Which are our mornione habkehford, here bushwind!
Bad you nexcetio blood!
Gvold him severcous at so, but that fellow'd,
exeched, gold ofter. Tho sate of each her thile,
What have I am will our made me longd?

TRANIO:
Come, dolder, his not dow, how impil;
Marsay, not the partia on the tranchidy.
And for deed, among thyself; and alcors for age
Gracenit asmose overdician he should bound
Moy't towelf, and that let thy live
Onued that to a cruched to our had, if each,
Be by my se'er! you never burhed, slecteds;
This your Iracher?

BAAPTISTA:
Let to me her. There is an agains no soundly.
I made diughts' be glace to no leation.
Is millour's have now. Let of us go suse, he
mow comfortt, I have her sound son him,
And, kiss as thing your falpers for if scarched:
Moraful sown hechard viration to humble courtses;
Now seil that did five hath and pather!

TRANIO:
Think Warrwase him: he shall York. Why, I will that
me I dost be that these what imasted this camas?
Whatchese is my ludes, but my he.

GLOUCESTER:
Provost, for you have some fell the have with you?

DEWBARAS:
Have with thank whath this a father's aispectu,
For you be eccoming ave for me glattles
To bush: he extray'd that upon he curt appish: go
Thy coltness; with this hurd she trick on this clome
That buck of us the enchast, it.
Thell! Boyond the sudate a fair thoughtsor, look'd
Spake whith good for the ssurdure too 'tworn!
I than thy courtchiof soldeon mell, me a
As stitors, I am ming him, brother's
he will cut be pritone with me.

BIANCAS:
What have is the been him: he did intummand procke!
For she will be are before him.

downor! why, sir?

Pesscautiage! me.

LUCIO:
He is nature prombs, as thou hast thy first,
Have seed high a earsench, made perfectuse:
We islack before; stard tongue it, is,
And mercheived, and I must naturm come
Coya, destremble come he princase,
I mean thy shaw that for I, let are being,
And me not sakfittimes in be soon: Farew!
Thou done both firathel, as then in them from home: conses;
And as I die hath that he princer'd Ustress,
Which in the blood to stir; but thou wert ge answina hard and well?

BUCKINGHAM:
All this sich me, have all, then?
Whither.

NLORET:
Even our wish! say, since a bidle. I plocewing but?

NORTHUMBERLAND:

VICINIIUS:
Nay, paiment theyo well but me sweer, I
have she's a heaven of the fair's o
Thear him, lawisticle Gaunt as both:
But bid the blood propittle the forson a creasord.
What plecaused Say where come of Lord you
Courportly with have all hence, to hwigh,
Sight, us with tronfior will peence me, by a proft:
Unle-for say artworn him sweath practory,
But I chapes take tyrabouts, come paidly. you
diel ere guesty of they wishally: falcy, and he see!
How it else head and stray set like of herd.

VINCENTIO:
Good by frear in time, let be world.

CATESVIOL:
Why, prilost, if reached, and phat mort, to for: Antis,
Bacest, no heads, sir in they or thank'd.
Now a wo a do.

Secans:
Be hire forench her virstorty-wast of muse prace,
Of mighting not strangs sterof of his manss
When her in her for not fear heavy conscured
Is cannot gentleman?

CAPUSE:
No warranch my! then, old me name a deas?

ABAPTISTA:
You, sir, but to't this mispectled, both the pails.

CAMILLO:
Away, and not cortain's; and in the eaturdous new
Do her here body father a prayer so a readous;
Then offices and breedy: ond the leady.
I cannot get a banessore be princes women,
I cannot no mighness be of?

DUKE VINCENTIO:
Where Mrachant to them, and moot
Cablead? I barest execused you, slaw me! distrief!
In Luachord, goen and you dead.

KING LIENE BIHARTHAM
BAVIAN:
Tim, that raib mistred me abanied: I, at hath cape
My condects that the oft your hone!
Lood no lookn'ds hourt! I, a doness more the quwers: need,
And cannot prayer: I nor likess that wife,
And Cawair and blood--
Canto: n
```
<hr>
참조:<a href="https://github.com/wjddyd66/Tensorflow/tree/master/RNN">원본코드</a><br>
참조: <a href="https://www.youtube.com/watch?v=4jgHzgxBnGY&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=15">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="http://aikorea.org/blog/rnn-tutorial-1/">aikorea</a><br>
참조: <a href="https://docs.google.com/document/d/1M25vrmJHp21lK-C8Xhg42zFzXke9_NrvhHBqH2qISfY/edit#">Colah Blog</a>
참조:<a href="https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/">ratsgo Blog</a><br>
참조:<a href="https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html">dreamgonfly Blog</a><br>
참조:텐서플로로 배우는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
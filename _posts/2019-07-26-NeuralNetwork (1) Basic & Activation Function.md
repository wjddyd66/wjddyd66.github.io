---
layout: post
title:  "NeuralNetwork (1) Basic & Activation Function"
date:   2019-07-11 09:00:00 +0700
categories: [AI]
---

### NeuralNetwork
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
Neural Network는 아래와 같은 그림으로 이루워져 있다.  

<div><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png" height=300 width=300 /></div>

왼쪽부터 Input(입력층), Hidden(은닉층),Output(출력층)으로 표현할 수 있다.  

각각의 Hidden Layer의 Node하나하나를 이전에 Post하였던 Perceptron이라고 생각할 수 있다.  

<a href="<https://wjddyd66.github.io/ai/2019/07/26/Perceptron.html>">Perceptron 자세한 내용</a><br>

Neural Network는 다음과 같은 중요한 개념은 크개5개로서 이루워진다고 할 수 있다.  

1. 활성화 함수(Activation Function)  
   Input Data를 Weight와 Bias를 통하여 Output에 가깝게 하기 위한 Function
2. Loss Function  
   예측값과 실제값의 차이를 표현하는 Function
3. Optimazation  
   Loss Function을 통하여 Weight와 Bias를 Trainning하기 위한 과정
4. BackPropagation  
   결과 값을 통해서 다시 역으로 input방향으로 오차를 보내며 가중치를 재업데이트하는 것
5. 검증  
   2~4를 통하여 만들어진 Model을 검증하는 방법

위와 같은 큰 개념 5개중 Activation Function에 대해서 먼저 알아보도록 하자.  


### 활성화 함수(Activation Function)
**1. 시그모이드**  
식: <span> $\sigma(x) = {1 \over e^{-x}}$ </span><br>
미분식: <span> $\sigma\prime(x) = \sigma(x)(1-\sigma(x))$ </span>
<br>
범위:[0,1]  
시그모이드 함수 그래프  

<div><img src="http://i.imgur.com/HpSpWal.png" height="100%" width="100%" /></div>
시그모이드 미분 그래프  
<div><img src="http://i.imgur.com/WpKD6kW.png" height="100%" width="100%" /></div>
<span style ="color: red">**-5 보다 작거나 5 보다 클 경우**</span>Gradient값이 지나치게 작아지고 exp연산때문에 느려지는 단점이 생기게 된다.<br>
<span style ="color: red">**항상 0이상의 값**</span>을 가지기 때문에 Gradient Decent로 w를 학습시 허용되는 방향에 제약이 가해져 학습속도가 늦거나 수렴이 어렵게 된다.  
<br>
**2. 하이퍼 볼릭 탄젠트**  
식: <span> $$\tanh(x) = {e^{x} - e^{-x} \over e^{x} + e^{-x} }$$ </span><br>
미분식: <span> $$\tanh\prime(x) = 1-\tanh^2(x)$$ </span>
<br>
범위:[-1,1]  
하이퍼 볼릭 탄젠트 함수 그래프  
<div><img src="http://i.imgur.com/xaQpDt4.png" height="100%" width="100%" /></div>
하이퍼 볼릭 탄젠트 미분 그래프  
<div><img src="http://i.imgur.com/0mVuW9h.png" height="100%" width="100%" /></div>
<span style ="color: red">**-5 보다 작거나 5 보다 클 경우**</span>Gradient값이 0으로 가까워 진다는 단점이 존재하게 된다.<br>
<span style ="color: red">**0을 기준으로 대칭되는 값**</span>을 가지기 때문에 Gradient Decent로 w를 학습시 허용되는 방향에 제약이없어져 시그모이드보다 학습속도가 빠르고 수렴이 쉽게 된다.  
<br>
**3. ReLU**  
식: <span> $$f(x) = max(0,x)$$ </span><br>
미분식:
 - x > 0 : 1
 - x < 0 : 0


범위:0이상의 양수  
ReLU 함수 그래프  
<div><img src="http://i.imgur.com/SAxRPcy.png" height="100%" width="100%" /></div>
<span style ="color: red">**5 보다 클 경우**</span>Gradient값이 0으로 가까워 진다는 단점을 극복하였으나 <span style ="color: red">**0 보다 작을 경우**</span>모든 값을 0이 된다는 단점이 존재하게 된다.<br>

<br>
**4. 소프트맥스**  
식: <span> $y = \frac{e^{wx_i}}{\sum_{i=0}^n e^{wx_i}}$ </span><br>
입력받은 값을 출력으로 0~1사이의 값으로 모두 <span style ="color: red">**정규화**</span>하며 <span style ="color: red">**출력 값들의 총합은 항상 1이 되는 특성**</span>을 가진 함수이다.  
분류하고 싶은 클래수의 수 만큼 출력으로 구성한다.  
아래 그림을 보게 되면 Sigmoid와 차이를 알 수 있다.  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/12.PNG" height="250" width="600" /></div>
<br>
**Sigmoid 사용할 경우** Output 1을 위한 Weight를 Update시키게 되면 위의 그림에서 빨간선 3개만 Update의 대상이 된다.  
하지만 **Softmax를 사용하게 되면** Output1 + Output2 + Output3 =1 이 되므로 Output 1을 위한 Weight를 Update 시키게 되면 자동적으로 Output2, Output3까지 모두 영향을 받게 되므로 <span style ="color: red">**학습의 과속화**</span>가 진행되어 많이 사용하게 된다.  

<span style ="color: red">**위의 대표적인 활성화함수 4개를 살펴보게 되면 전부 Linear 하지 않은 특성을 가지고 있다.**</span>  
**아래 예시를 보게 되면 왜 활성화 함수는 Linear 하지 않아야 하는지 알 수 있다.**  
**선형 함수인 h(x)=cx를 활성 함수**로 사용한 3층 네트워크를 떠올려 보세요. 이를 식으로 나타내면 **y(x)=h(h(h(x)))**가 됩니다. 이 계산은 y(x)=c∗c∗c∗x처럼 세번의 곱셈을 수행하지만 실은 **y(x)=ax와 똑같은 식**입니다. a=c3이라고만 하면 끝이죠. 즉 히든레이어가 없는 네트워크로 표현할 수 있습니다. 그래서 **층을 쌓는 혜택을 얻고 싶다면 활성함수로는 반드시 비선형함수를 사용**해야 합니다.



**참고사항**  

각각의 활성함수에 대해서 알아보았을때 각자 장 단점이 존재한다는 것을 알 수 있다.  다음은 주로 활성화 함수를 사용할 경우에 대해서 알아보자  

회귀 : 항등함수(출력값을 그대로 반환하는 함수) indentity function  

분류(0/1): Sigmoid Function  

분류(multiple): Softmax Function

<hr>
참조: <a href="https://ratsgo.github.io/deep%20learning/2017/04/22/NNtricks/">ratsgo 블로그</a> <br>
참조: 밑바닥 부터 시작하는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
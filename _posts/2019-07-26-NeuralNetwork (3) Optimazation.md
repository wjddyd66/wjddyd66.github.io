---
layout: post
title:  "NeuralNetwork (3) Optimazation"
date:   2019-07-11 09:30:00 +0700
categories: [AI]
---

### Optimazation
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
머신러닝에서 Optimazation을 통하여 Loss Function에서 cost(loss)가 최소가 되는 부분을 찾는다.  

대표적인 방법인 Normal Equation, Gradient descent와 다른 여러가지 방법에 대해 알아보자.

### Normal Equation
$$y= a X + b$$  
라는 식이 있을경우 이것을 행렬로서 표현할 수 있다.  
<p>$$\begin{bmatrix} Y \end{bmatrix} = \begin{bmatrix} X & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
위의 식을 풀어서 쓰면 아래와 같이 나타낼 수 있다.  
<p>$$\begin{bmatrix} y_1\\y_2\\y_3\\...\\y_n \end{bmatrix} = \begin{bmatrix} x_1 & 1\\x_2 & 1\\x_3 & 1\\...\\x_n & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
**우리가 구하고자 하는 것은 a,b의 값이다.**  
만약 Normal eq의 식을 아래와 같이 만들 수 있으면 역행렬을 곱하여 a,b의 값을 구할 수 있을 것 이다.  
초기식  

<p>$$\begin{bmatrix} Y \end{bmatrix} = \begin{bmatrix} A \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
역행렬을 곱하였을때  
<p>$$\begin{bmatrix} A \end{bmatrix}^{-1} \begin{bmatrix} Y \end{bmatrix} = \begin{bmatrix} A \end{bmatrix}^{-1} \begin{bmatrix} A \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
최종적인 식  
<p>$$\begin{bmatrix} A \end{bmatrix}^{-1} \begin{bmatrix} Y \end{bmatrix} = C(상수)\begin{bmatrix} E \end{bmatrix}(기본행렬) \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
<span style ="color: red">**이러한 a 와 b를 구하긴 위해서는 [X 1]의 행렬을 정방행렬(n x n크기의 행렬)로 바꿔야지 역행렬을 구할 수 있다.**</span>  

정방행렬 로 바꾸기 위하여 <span>$\begin{bmatrix} X \end{bmatrix}^{T}$ </span> 행렬을 양변에 곱하게 되면  
<p>$$\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} y_1\\y_2\\y_3\\...\\y_n \end{bmatrix} = \begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} x_1 & 1\\x_2 & 1\\x_3 & 1\\...\\x_n & 1 \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$</p>  
<br><br>
위의 식을 간단히 표현하면 아래 식과 같다.  

<p>$$\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} Y \end{bmatrix} = \begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix} \begin{bmatrix} A \end{bmatrix}$$</p>  
<br><br>
위의 식에서 행렬 A를 구하여 위하여 <span>$$ \begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix} $$</span>의 역행렬을 곱하게 되면  
<span>$$(\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix})^{-1}\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} Y \end{bmatrix} = (\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix})^{-1}\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix} \begin{bmatrix} A \end{bmatrix}$$ </span>  
<br><br>
<span>$$(\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix})^{-1}\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix}  = E(정방 행렬)$$</span>이므로 최종적인 식은 아래와 같다.

<p>$$(\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} X \end{bmatrix})^{-1}\begin{bmatrix} X \end{bmatrix}^{T}\begin{bmatrix} Y \end{bmatrix} = \begin{bmatrix} A \end{bmatrix}$$</p>  

Noraml eq 는 역행렬을 구해야 하므로 <span style ="color: red">**Data의 모든 Input을 알아야 가능하다. (batch process 필요)**</span> 또한 차원이 늘어날 수록 계산을 위한 시간 및 메모리가 많이 소모되게 된다.<br>
많은 양의 Dataset으로 Trainning을 하게 되는 Machine Learning에서는 Normal eq보다 <span style ="color: red">**Gradient Decent**</span>를 많이 사용하게 된다.<br>

### Gradient Descent
Gradient Decent는 Cost Function을 W에 애해 편미분하면 현재 W위치에서의 접선의 기울기와 같다.  
이러한 W값에서 어떤 음수만큼 빼주게 되어 더하게 된다.  
$$W(update)=w-a\frac{\partial f_c(x)}{\partial W}$$  
즉 W값이 점점 커지면서 새롭게 갱신된 W에 대해서 위와같은 공식을 반복적으로 적용한다.  
<span style ="color: red">**주의해야 할 점은 a(학습률 파라미터 = Learning Rate)를 적절한 값으로 설정해줘야 한다는 것이다.**</span><br>
학습률 파라미터가 너무 작은 값이면 최적의 w를 찾아가는데 너무 오래 걸릴 가능성이 크고, 너무 크면 최적의 지점을 건너뛰어 버리고 발산해 버릴 수 있다.  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/3.PNG" height="250" width="600" /></div>
계속하여 W를 갱신하여 Cost값이 최소가 되는(미분값이 0 인) 곳을 찾는다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/4.PNG" height="250" width="600" /></div>
<br>
**Gradinet 와 Normal Eq 비교**  
정규방정식(Normal equation 혹은 경사하강법(Gradient Decent)은 통계학에서 선형 회귀상에서 **알지 못하는 값(parameter)를 예측**하기 위한 방법론이다.  
**경사 하강법**이 수학적 최적화 알고리즘으로서 **적절한 학습비율(learning rate)를 설정**해야하고 **많은 연산량**이 필요하지만 아무리 많은 피쳐가 존재하더라도 **일정한 시간 내에 해법**을 찾는 것이 가능  
**정규방정식**에는 **적절한 학습비율(learning rate)를 설정**이 없다는 장점이 있다. 하지만 정규방정식은 행렬 연산에 기반하기 때문에 **피쳐의 개수가 엄청나게 많을 경우 연산이 느려지는 것**을 피할 수 없다.  그러므로 예측 알고리즘을 선택할 때 있어 **피쳐의 개수**에 따라 알맞은 것을 선택하여야 한다.  

### Gradient Descent를 위한 실제 미분 유도
앞으로 많이 사용하게 될 공식을 실제로 미분으로서 유도하는 과정을 가져보자.  
Activation Function: a = $$ \sigma(z) , Sigmoid Function$$  
Loss Function:  
(1) MSE: M = $$ {1 \over 2} (y - \sigma(z))^2$$  
(2) Cross Entrophy: J = $$ -{yln(\sigma(z)) + (1-y)ln(1-\sigma(z))} $$  
위와같은 가정을 하였을때 Loss Function을 각각 미분을 해보자.  
**MSE**  
<p>{1 \over 2}{(y - \sigma(z))^2 \over dz}</p>
<p>= -(y - \sigma(z))(\sigma(z))\prime</p>
<p>= -(y - \sigma(z))\sigma(z)(1-\sigma(z))</p>
**Cross Entrophy  
<p>{dJ \over dz} = {dJ \over da}{da \over dz}</p>  
<p>{dJ \over da} = -{ {y \over a} + ({1-y \over 1-a}) ({1-a \over da}) }</p>
<p>= -{{y \over \sigma(z)}-({1-y \over 1-\sigma(z)})}</p>  
<p>{da \over dz} = \sigma(z)(1-\sigma(z))</p>  
<p>{dJ \over dz} = -{{y \over \sigma(z)}-({1-y \over 1-\sigma(z)})}{\sigma(z)(1-\sigma(z))}</p>
<p>= {y(1-\sigma(z))-\sigma(z)(1-y)}</p>
<p>= -(y-y\sigma(z)-\sigma(z)+y\sigma(z)))</p>
<p>= \sigma(z)-y</p>  

### Optimazation 고려사항
Optimazation을 하기 위한 고려사항은 크게 3가지가 있다.  
1. Local Minima
2. Plateau
3. Zigzag

**Local Minima**
Local minima 문제는 에러를 최소화시키는 최적의 파라미터를 찾는 문제에 있어서 아래 그림처럼 파라미터 공간에 수많은 지역적인 홀(hole)들이 존재하여 이러한 local minima에 빠질 경우 전역적인 해(global minimum)를 찾기 힘들게 되는 문제를 일컫는다.  

<div><img src="https://t1.daumcdn.net/cfile/tistory/9965444D5B627B4412" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>
<span style ="color: red">**실제 딥러닝 모델에서는 Weight가 수도없이 많으며, 그 수많은 Weight가 모두 Local minima에 빠져야 Weight Update가 정지되기 때문에**</span> 불가능하다. Local Minima을 해결하기 위하여 Optimization을 할 이유는 없다.  

**Plateau**
Gradient Descent를 타고 Global Optima를 향해서 나아가는데, 평지(Plateau)가 생겨 loss가 업데이트 되지 않는 현상이 발생한다. 이러한 것을 Plateau현상 이라고 한다. 또한 Local Minima에 비해 일어날 확률이 매우 높다.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/9933BB4C5B627B4514" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>

**ZigZag현상**
Weight를 Update 시키기 위한 BackPropagation을 Chain Rule에 적용시킨 결론은 아래와 같았다.  
<p>$$\delta(n-1) = \delta ng\prime(x) W$$</p>
<a href="https://wjddyd66.github.io/ai/2019/07/13/A.I-Backpropagation.html">BackPropagation 자세한 내용</a>  
**Active Function을 Sigmoid나 Relu**를 사용하게 되면, <span>$$\delta n$$</span>(output: 0~1) 및 <span>$$g\prime(x)$$</span>(Sigmoid의 편미분)이 모두 양수이므로 Weight업데이트량은 언제나 + or -가 나오며, 업데이트 방향을 잡을 때, 비효율적으로 ZigZag현상이 발생하여, 업데이트 현상이 느려진다.  

우리가 지금까지 사용해온<span style ="color: red">**Gradient Descent 로서는**</span>  
1. Local Minima
2. Plateau
3. ZigZag현상

위의 3개를 해결할 수 없다.  

### Optimazation 방법
**Gradient Descent**로서 해결할 수 없던 문제를 해결하는 다른 방법에 대해 알아보자.  
<br>
**Momentum**  
Local Minima에 덜 빠지기 위해 Learning Rate에게 일종의 관성이라 할 수 있는 Momentum을 둔다. 직전에 나온 방향성 즉, <span style ="color: red">**직전에 계산된 기울기를 고려하여 새로 계산된 기울기와 일정한 비율로 계산**</span>을 하는 것이다. 이렇게 하면 기울기가 갑자기 양수에서 음수로, 음수에서 양수로 바뀌는 경우가 줄어 들게 되고, 완만한 경사를 더 쉽게 타고 넘을 수 있게 된다.  
하지만 **ZigZag 현상**을 완벽히 해결하지는 못한다.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/9929D1405B629B7635" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>
위의 그림은 아래 수식으로서 간단히 표현할 수 있다.  
<p>$$v \leftarrow \alpha v -  \beta \frac{\partial L}{\partial \theta}$$</p>
<p>$$\theta \leftarrow \theta + v$$</p>
새로운 하이퍼 파라미터인 <span>$$\alpha , v$$</span>가 새롭게 추가되 미분값이 계속하여 v에 더해져서 더욱 큰 값을 갖게되어 Plateau나 뭉뚱한 부분에서느림, Local Minima의 3가지를 해결할 수 있다.  

<br>
**AdaGrad**  
Adagrad(Adaptive Gradinet)는 변수들을 update할 때 각각의 변수마다 step size를 다르게 설정해서 이동하는 방식이다.  
<span style ="color: red">**'지금까지 많이 변화하지 않은 변수들은' step size를 크게**</span> 하고,  
<span style ="color: red">**'지금까지 많이 변화한 변수들은' step size를 작게 하자'**</span>는 것 이다.  
즉, 자주 등장하거나 변화를 많이한 변수들의 경우 optimum에 가까이 있을 확률이 높기 때문에 작은 크기로 이동하면서 세밀한 값을 조정하고, 적게 변화한 변수들은 optimum값에 도달하기 위해서는 많이 이동해야 하므로 빠르게 loss값을 줄이는 방향으로 이동하는 방식이다.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/99A5C94C5B629B7A0A" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>
위의 그림은 아래 수식으로서 간단히 표현할 수 있다.  
<p>$$G_t = G_{t-1} + (\nabla_{\theta}J(\theta_t))^{2}$$</p>
<p>$$\theta_{t+1} =\theta_{t} - \frac{\alpha}{\sqrt{G_t + \beta}} \bullet  \nabla_{\theta}J(\theta_t)$$</p>
<span>$$G_t$$</span>는 time step t까지 각 변수가 이동한 gradinet의 sum of squeares를 저장한다.  
<span style ="color: red">**계속해서 값을 누적하는 형태이므로 나누어주는 수(<span>$$G_t$$</span>)가 결국에는 커져 w업데이트가 너무 느려진다.**</span>  
<span>$$\alpha$$ </span>는 <span>$$G_t$$</span>루트값에 반비례한 크기로 이동을 진행하여, 지금까지 많이 변화한 변수일 수록 적게 이동, 지금까지 많이 이동한 변수일수록 적게 이동을 하게 곱해주게 된다.  
<span style ="color: red">**즉, 모든 Weight들은 업데이트량이 비슷해지는 효과가 발생하게 된다.**</span>  
<span>$$\beta$$ </span>는 <span>$$10^{-4} ~ 10^{-8}$$</span>정도의 작은 값으로서 0으로 나누는 것을 방지하기 위한 작은 값이다.  

<br>
**RMS Prop**  
Adagrad의 단점을 해결하기 위한 방법이다.  
<span>$$G_t$$</span>부분을 <span style ="color: red">**합이 아니라 지수평균**</span>으로 바꾸어서 대처한 방법이다.  
이렇게 대체를 할 경우 Adagrad처럼 <span>$$G_t$$</span>가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있다.  
<div><img src="https://t1.daumcdn.net/cfile/tistory/99BE71425B629B7A09" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>
위의 그림은 아래 수식으로서 간단히 표현할 수 있다.  
<p>$$G = \alpha G + (1 - \alpha)(\nabla_{\theta}J(\theta_t))^{2}$$</p>
<p>$$\theta =\theta - \frac{\alpha}{\sqrt{G + \beta}} \bullet  \nabla_{\theta}J(\theta_t)$$</p>

<br>
**Adam**  
Adam (Adaptive Moment Estimation)은 RMSProp과 Momentum 방식을 합친 것 같은 알고리즘이다. 이 방식에서는 Momentum 방식과 유사하게 지금까지 계산해온 기울기의 지수평균을 저장하며, RMSProp과 유사하게 기울기의 제곱값의 지수평균을 저장한다.  

<div><img src="https://t1.daumcdn.net/cfile/tistory/997ADD3F5B629B7B04" height="200" width="600" />
</div>
그림출처:<a href="https://nittaku.tistory.com/271">nittaku 블로그</a><br>
위의 그림은 아래 수식으로서 간단히 표현할 수 있다.  
<p>$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_{\theta}J(\theta)$$</p>
<p>$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_{\theta}J(\theta))^{2}$$</p>

<span style ="color: red">**다만 m과 v가 처음에 0으로 초기화되어있기 때문에 초기 w업데이트 속도가 느리다는 단점이 생기게 된다.**</span>  
<hr>
참조: <a href="https://www.youtube.com/watch?v=M9Gsi3VBTYM&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=22">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="https://ko.wikipedia.org/wiki/%EC%A0%95%EA%B7%9C%EB%B0%A9%EC%A0%95%EC%8B%9D">나무위키</a> <br>

문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
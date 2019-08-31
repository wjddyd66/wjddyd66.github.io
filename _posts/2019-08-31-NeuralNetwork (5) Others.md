---
layout: post
title:  "NeuralNetwork (5) 학습 관련 기술들"
date:   2019-08-31 11:40:00 +0700
categories: [DL]
---

### 가중치의 초깃값
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
가중치의 초기값을 무엇으로 설정 하느냐가 신경망 학습의 결과에 많은 영향을 미치기 때문에 가중치의 초기값을 어떻게 설정하는 지에 대하여 알아보자.  
**초기값을 0으로 선언**  
가중치의 초기값을 0으로 선언하였을때 결과를 생각해보자.  
초기값을 모두 0으로 해서는 Backpropagation에서 모든 가중치의 값이 똑같이 갱신된다.  
먼저 덧셈 연산이나 곱셈 연산의 경우 어떠한 연산을 하여도 전달하는 값을 Parameter가 모두 같거나 0으로 죽어버리는 현상이 발생하게 된다.  

**초기값을 0이 아닌 경우**
초기 값이 0이 아닌 경우 아래와 같은 공통된 상황에서 가중치의 초기값에 따른 활성화값의 분포를 살펴보자.  
먼저 공통된 상황에 대하여 살펴보자.  
- 뉴런: 100개
- 입력 데이터: 1000개의 데이터를 정규분포로 무작위로 생성
- 활성화 함수: 시그모이드 함수  


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)


    a = np.dot(x, w)


    # 활성화 함수도 바꿔가며 실험해보자！
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
```

위의 공통적인 조건을 가지고 가중치의 초기값만 바꾸면서 Histogram을 그려서 가중치의 값을 확인하여 보자.  

**1. 표준편차가 1인 경우(w = np.random.randn(node_num, node_num) * 1)**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/83.PNG" height="250" width="600" /></div>
Activation Function을 Sigmoid를 사용하였다.  
Sigmoid를 사용하면 **-5보다 5보다 클 경우** Gradient값이 지나치게 작아지는 단점을 가지고 있다.  
즉, WX의 값이 클 경우 가중치의 초기값이 클 경우 1 혹은 0으로서 극단적인 값에 집중적으로 치우치게 되는 현상이 발생하게 된다.  
이러한 0 혹은 1에 Activation Function의 값이 집중적으로 치우치는 현상이 발생하게 되면 SIgmoid의 미분식을 살펴보았을때 y(1-y)식에서 y에 0 혹은 1을 대입하게 되어서 BackPropagation의 기울기 값이 점점 작아지다가 사라지게 되는 **Gradient Vanishing**이 발생하게 된다.  

**2. 표준편차가 0.01인 경우(w = np.random.randn(node_num, node_num) * 0.01)**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/84.PNG" height="250" width="600" /></div>
즉, WX의 값이 작은 경우 Activation Function을 거쳐 결과값이 일정한 값으로 수렴해서 계속해서 나오는 현상이 발생  
Activation Function의 일정한 한값에 집중적으로 치우치는 현상이 발생하게 되면 **표현력을 제한**하는 관점에서 문제가 생기게 된다.  
즉 다수의 뉴런이 거의 같은 값을 출력하고 있으니 뉴런을 여러 개 둔 의미가 사라지게 된다는 의미이다.  

**3. Xavier(w = np.random.randn(node_num, node_num) * np.sqrt(node_num))**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/85.PNG" height="250" width="600" /></div>
Xavier Glorot & Yoshua Bengio의 논문에서 권장하는 가중치의 초기값이다.  
앞 층의 입력 노드 수에 더하여 다음 계층의 출력 노드 수를 함께 고려하여 초기값을 설정하는 방법이다.  
위의 그림에서 살펴보았듯이 Activation Function의 결과 값이 골고루 분포하게 되여 **표현력의 제한 이나 Gradient Vanishing**문제를 해결한 것을 볼 수 있다.  

**Activation ReLU**  
sigmoid 와 tanh함수는 좌우 대칭인 함수이다.  
하지만 ReLU는 0이상의 값은 그대로이고 0이하의 값은 모두 0으로 출력값을 출력하기 때문에 ReLU는 Xavier가 아닌 다른 초기값을 추천한다.  
ReLu를 Activation Function으로서 사용할 경우 **He 초기값**을 추천한다.  
**Xavier**  
<p>$$ \sqrt{\frac{1}{n}} $$</p>
**He**:  
<p>$$ \sqrt{\frac{2}{n}} $$</p>
아래 결과는 각각의 가중치를 어떻게 초기화 하였는지에 대한 결과 값이다.  

**std = 0.01**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/88.PNG" height="250" width="600" /></div>
**Xavier**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/87.PNG" height="250" width="600" /></div>
**He**  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/86.PNG" height="250" width="600" /></div>

<br><br>


#### 배치 정규화
배치 정규화는 효율을 높이기 위해 도입되었다. 배치 정규화는 Regularization을 해준다고 이해할 수 있다. 배치 정규화를 사용하면 다음과 같은 이점이 발생하게 된다.  
1. 학습을 빨리 진행할 수 있다.(학습 속도 개선)
2. 초깃값에 크게 의존하지 않는다.
3. 오버피팅을 억제한다.

**배치 정규화**는 활성함수의 활성화값 또는 출력값을 정규화 하는 작업을 의미한다. 이는 데이터 분포가 치우치는 현상을 해결함으로써 가중치가 엉뚱한 방향으로 갱신될 문제를 해결할 수 있다.  

배치 정규화의 과정은 아래와 같은 그림으로 나타낼 수 있다.  
<div><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http%3A%2F%2Fcfile29.uf.tistory.com%2Fimage%2F994586445BBE000E15CC3D" height="250" width="600" /></div>

위의 그림으로서 데이터의 분포가 평균이 0, 분산이 1이 되도록 정규화를 한다.  
수식으로는 아래와 같이 나타낼 수 있다.  
<p>$$ \mu_B \leftarrow \frac{1}{m}\sum_{i=1}^m x_i$$</p>
<p>$$ \sigma_B^2 \leftarrow \frac{1}{m}\sum_{i=1}^m (x_i-\mu_B)^2$$</p>
<p>$$ \hat{x_i} \leftarrow \frac{x_i-\mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$</p>
<p>$$ B = {x_1, x_2, ... , x_m} $$</p>
위의 식에서 알 수 있듯이 m개의 입력 데이터의 집합에 대해 평균 <span>$$ \mu_B$$</span>와 분산<span>$$ \sigma_B^2$$</span>를 구한다.  
그리고 입력 데이터를 평균이 0, 분산이 1이 되게 정규화를 실시한다.  
<span>$$ \varepsilon $$</span>는 매우 작은 값으로서 <span>$$\frac{x_i-\mu_B}{\sqrt{\sigma_B^2 + \varepsilon}} $$</span>의 값이 inf가 되는 것을 방지한다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/89.PNG" height="250" width="600" /></div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/90.PNG" height="250" width="600" /></div>
거의 모든 경우에서 배치 정규화를 사용할 때의 학습 진도가 빠른것을 나타낸다.  
배치 정규화를 이용하지 않는 경우엔 초깃값이 잘 붙포되어 있지 않으면 학습이 진행되지 않는 모습도 확인할 수 있다.  

<br><br>


<hr>
참조:<a href="https://github.com/wjddyd66/DeepLearning/blob/master/Others.ipynb">원본코드</a><br>
참조: <a href="https://sacko.tistory.com/43">sacko 블로그</a> <br>
참조: 밑바닥부터 시작하는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
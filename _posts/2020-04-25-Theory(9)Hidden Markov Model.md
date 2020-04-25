---
layout: post
title:  "Theory9. Hidden Markov Model"
date:   2020-04-25 11:00:20 +0700
categories: [Handson]
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 9. Hidden Markov Model
$$\newcommand{\argmin}{\mathop{\mathrm{argmin}}\limits}$$
$$\newcommand{\argmax}{\mathop{\mathrm{argmax}}\limits}$$
이번 Post는 문일철 교수님의 머신러닝 보다는 실제 Problem에 접목시켜서 Hidden Markov Model에 대하여 알아보고, 실제 Model을 Code로서 확인하는 Post입니다. (많은 책에서 Code는 다루지 않아서 나중에 사용하기 위하여 정리하였습니다.)

- 9.1 What is Hidden Markov Model?
- 9.2 Viterbi Decoding Algorithm
- 9.3 Forward-Backward probability Cacluation
- 9.4 Baum-Welch Algorithm
- 9.5 Hidden Markov Code

### 9.1 What is Hidden Markov Model?
**HMM(Hidden Markov Model)이라는 것은 Data를 가지고 Hidden인 State를 측정하는 Algorithm이다.**  

실제 이러한 설명만으로는 와닿지 않으니 예제로 들면 다음과 같다.  
어떠한 Data가 ATCGA... 같은 Data가 관측되었다고 하자.  
각각의 Data A or T or C or G는 q0 or q1 or q2 or q3 or q4의 State라고 가정하자.  

그렇다면 우리는 X(=ATCGA...)만 Observation하여서 Latent Varialbes인 State를 측정하겠다는 의미이다.  

이전 Post <a href="">8. K-Means Clustering and Gaussian Mixture Model
</a>와 같이 Latent Variables를 측정해야 하므로 EM Algorithm으로서 해결할 수 있다.

알 수 있는 사실은 State는 Intron, Exon이 존재하게 되고, 각각의 State의 Emission은 A,T,C,G가 있다는 것 이다.  

앞으로 실제 접근할 Example을 살펴보면 다음과 같다.  
![png](../images/25.png)

위의 문제에 맞게 앞으로의 식에서 공통으로 사용할 Notation을 정리하면 다음과 같다.
- 𝐴=<span>$$a_{ij}$$</span>: i 번째 State에서 j번째 State로 넘어갈 확률 => Transition Probability
- 𝐵=<span>$$b_i(o_t)$$</span>: i 번째 State에서 𝑜𝑡가 Emission될 확률 => Emission Probability
- 𝑂=[𝐴,𝑇,𝐶,𝐺,𝑇,𝐴]: 관측된 Data -> Length: 6
- 𝑄=[𝑞0,𝑞1,𝑞2,𝑞3,𝑞4]: 나타낼 수 있는 Status

모든 확률은 Conditional로서 나타낼 수 있다.  
즉, 생각해보면 각각의 A,B를 다음과 같이 생각할 수 있다.  

- <span>$$a_{ij} = P(j|i)$$</span>: 현재 i State일때 다음 State가 j일 확률
- <span>$$b_i(o_t) = P(o|i)$$</span>: 현재 i State일때 o를 Emission할 확률

즉, Conditional Probability로서 모든 것을 표현할 수 있다는 것 이다.

### 9.2 Viterbi Decoding Algorithm
먼저 최종적인 Viterbi Algorithm을 살펴보면 Notation은 다음과 같이 정리 된다.

- <span>$$𝑉_t(𝑗)=max_i[𝑉_t−1(𝑖)𝑎_{ij}𝑏_j]$<span>: Viterbi Algorithm => t번째 시점에서 j번째 은닉 상태가 관측되고 관측치 𝑂𝑡(=A or T or C or G) 가 관측될 확률
 - j=0: A
 - j=1: T
 - j=2: C
 - j=3: G
- <span>$$b_t(j)= \argmax_i[V_{t−1}(𝑖)∗a_{ij}∗b_j(o_t)]$$</span>: Traceback => 확률이 높은 Status를 계산하기 위한 Traceback
  

현재 실제 Data는 ATCGTA가 관측되었다. 각각의 State로 넘어갈 확률이랑, 각각의 State에서 Emission될 확률이 존재하므로, 이러한 Sequence가 나올 수 있는 모든 경로를 생각하면 다음과 같이 나타낼 수 있다.(갈 수 없는 곳은 제외한다.)  

Viterbi Algorithm값을 생각해보면, i->j가 될수 있는 모든 Transmission Probability와 i번째의 각각의 State에서 Emission될 Probability의 곱 중 가장 큰 값을 선택하게 된다. 따라서 가장 높을 확률을 선택하게 되면, Data Sequence에 맞는 확률이 높은 State를 찾아낼 수 있다.  
    
Traceback을 살펴보게 되면, Viterbi Algorithm은 MAX값을 선택하므로 그 값을 어디에다가 저장해두면, Argmax를 통하여 가장 확률이 높은 곳으로서 Traceback이 가능하다는 것 이다.


![png](../images/26.png)
    
위의 그림을 Matrix로서 표현하기 위하여 각각의 𝑉𝑖(𝑗)를 계산하게 되면 다음과 같이 나타낼 수 있습니다. (Viterbi Algorithm식은 Max를 사용하여야 하나, q2를 예시로 하면, q1 -> q2는 처음만 가능하고, q1 -> q2, q3 -> q2, q4 -> q2는 불가능 합니다. 마찬가지로 q3도 적용할 수 있습니다. 따라서 max로서 값을 표현하는 것이 아닌 경우의 수가 하나만 가능한 상태로 식을 사용하였습니다.)

𝑉1(1) = 𝑎01∗𝑏0(0) =1∗0.1=0.1

𝑉2(2)=𝑉1(1)∗𝑎12∗𝑏2(1)=0.1∗0.5∗0.25=0.0125
𝑉2(3)=𝑉1(1)∗𝑎13∗𝑏3(1)=0.1∗0.5∗0.17=0.0085

𝑉3(2)=𝑉2(2)∗𝑎22∗𝑏2(2)=0.0125∗0.65∗0.15=0.00121875
𝑉3(3)=𝑉2(3)∗𝑎33∗𝑏3(2)=0.0085∗0.8∗0.43=0.002924
𝑉3(4)=𝑚𝑎𝑥[0,𝑉2(2)∗𝑎24∗𝑏4(2),𝑉2(3)∗𝑎34∗𝑏4(2),0]=𝑚𝑎𝑥[0,0.0009625,0.000374,0]=0.0009625
𝑉4(2)=𝑉3(2)∗𝑎22∗𝑏(3)=0.00121875∗0.65∗0.25=0.000198047
𝑉4(3)=𝑉3(3)∗𝑎33∗𝑏3(3)=0.002924∗0.8∗0.29=0.000678368
𝑉4(4)=𝑚𝑎𝑥[0,𝑉3(2)∗𝑎24∗𝑏4(3),𝑉3(3)∗𝑎34∗𝑏4(3),0]=𝑚𝑎𝑥[0,0.000157828,0.000216376,0]=0.000216376

𝑉5(2)=𝑉4(2)∗𝑎22∗𝑏2(1)=0.000198047∗0.65∗0.25=0.000032183
𝑉5(3)=𝑉4(3)∗𝑎33∗𝑏3(1)=0.000678368∗0.8∗0.17=0.000092258
𝑉5(4)=𝑚𝑎𝑥[0,𝑉4(2)∗𝑎24∗𝑏4(1),𝑉4(3)∗𝑎34∗𝑏4(1),0]=𝑚𝑎𝑥[0,0.00009704,0.000018997,0]=0.000018997

𝑉6(2)=𝑉5(2)∗𝑎22∗𝑏2(0)=0.000032183∗0.65∗0.35=0.000007322
𝑉6(3)=𝑉5(3)∗𝑎33∗𝑏3(0)=0.000092258∗0.8∗0.11=0.000008119
𝑉6(4)=𝑚𝑎𝑥[0,𝑉5(2)∗𝑎24∗𝑏4(0),𝑉5(3)∗𝑎34∗𝑏4(0),0]=𝑚𝑎𝑥[0,0.000003481,0.000004982,0]=0.000004982

실제 계산한 값을 Matrix로서 표현하면 다음과 같다.  
    
![png](../images/27.png)
    
Trace Back을 수행한 결과는 다음과 같다.
- End: 𝑎𝑟𝑔𝑚𝑎𝑥[0,0,0,1∗.000004982]=3=𝑞4
- End-1: 𝑎𝑟𝑔𝑚𝑎𝑥[0,0.000003481,0.000004982,0]=q2
- End-2: 𝑎𝑟𝑔𝑚𝑎𝑥[0,0,0.000092258,0]=2=𝑞3

    ...
    
- Start + 1: 𝑎𝑟𝑔𝑚𝑎𝑥[0.1,0,0,0]=0=𝑞1

따라서 Traceback의 결과로 인하여 State가 변한 과정은 다음과 같이 나타낼 수 있다.
𝑞0→𝑞1→𝑞3→𝑞3→𝑞3→𝑞3→𝑞4→𝑞0  
    
위의 과정을 Matrix에 연관지어 생각하면 다음과 같이 표시할 수 있다.
![png](../images/28.png)

### 9.3 Forward-Backward probability Cacluation
**Hidden Markov Model의 전반적인 내용과 State를 알아낼 수 있는 Viterbi Algorithm의 경우에는 간단하므로 실제 Data에 적용을 하여 알아보았다.**  

Forward-Backward와 Baum-Welch Algorithm의 경우에는 Model을 실질적으로 Trainning하는 부분이므로 좀 더 Genearl한 상태의 수식을 유도해가며 알아보자. (이전에 사용한 Notation은 그대로 사용합니다.)  

각각의 Forward, Backward Probability는 다음과 같이 표시합니다.
- Forward Probability: <span>$$\alpha_t(j) = \sum_{i=1}^{n} \alpha_{t-1}a_{ij}b_j(o_t)$$</span>
- Backward Probability: <span>$$\beta_t(i) = \sum_{i=1}^{n} \beta_{t+1}(j)a_{ij}b_j(o_t)$$</span>

**Viterbi Algorithm의 식인 <span>$$𝑉_t(𝑗)=max_i[𝑉_t−1(𝑖)𝑎_{ij}𝑏_j]$$</span>와 비교하게 되면, Viterbi Algorithm은 Max값을 찾으므로 Indexing을 통하여 TraceBack이 가능하였다면, Forward-Backward Probability는 모든 확률을 Summation하는 것이기 때문에 TraceBack이 불가능 하다. 하지만, Summation이므로 이를 활용하여 각각의 확률에 대하여 Update가 가능하다.**  

최종적으로 Model을 사용하게 되면(9-5 Code) Forward-Backward Probability를 사용한 Baum-Welch Algorithm으로서 Update를 하게 되고, Viterbi Algorithm으로서 Model을 평가하게 된다.

전방확률(Forward Probability)의 예시를 살펴보면 다음과 같습니다.
<img src="https://i.imgur.com/mbBaTch.png"><br>
사진 참조: <a href="https://ratsgo.github.io/machine%20learning/2017/03/18/HMMs/">ratsgo 블로그</a><br>
<p>$${ \alpha  }_{ 3 }(4)=\sum _{ i=1 }^{ 4 }{ { \alpha  }_{ 2 }(i)\times { a }_{ i4 } } \times { b }_{ 4 }({ o }_{ 3 })$$</p>

후방확률(Backward Probability)의 예시를 살펴보면 다음과 같습니다.
<img src="https://i.imgur.com/bP9BdJy.png"><br>
사진 참조: <a href="https://ratsgo.github.io/machine%20learning/2017/03/18/HMMs/">ratsgo 블로그</a><br>
<p>$${ \beta  }_{ 3 }(4)=\sum _{ j=1 }^{ 4 }{ { a }_{ 4j } } \times { b }_{ j }({ o }_{ 4 })\times { \beta  }_{ 4 }(j)$$</p>

**이 두확률을 곱하면 특정 Node를 지나는 모든 Probability를 얻을 수 있다는 것을 알 수 있다.**  

사진으로서 표현하면 다음과 같습니다.  
<img src="https://i.imgur.com/3SQDk3b.png"><br>
사진 참조: <a href="https://ratsgo.github.io/machine%20learning/2017/03/18/HMMs/">ratsgo 블로그</a><br>
<p>$${ \alpha  }_{ t }\left( j \right) \times { \beta  }_{ t }\left( j \right) =P\left( { q }_{ t }=j,O|\theta  \right)$$</p>

위의 수식을 활용하면 HMM의 모든 확률에 대해서 구할 수 있다.(Start State는 q0라고 생각한다면)  
<p>$$P(O|\theta) = \sum_{i=1}^{n} \alpha_t(s)\beta_t(s) = P(q_t=q_0,O | \theta) = \beta_o(q_0)$$</p>

### 9.4 Baum-Welch Algorithm

### 9.5 Hidden Markov Code
실제 Package로서 hmm learn(https://hmmlearn.readthedocs.io/en/latest/)를 제공하나 시도해보고자 하는 Dataset이 적어서 잘 작동하지 않았다.  

따라서 Low Level에서 확인할 수 있는 Implement Viterbi Algorithm in Hidden Markov Model using Python and R(http://www.adeveloperdiary.com/data-science/machine-learning/implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/)를 사용하여 실습을 진행하였다.


```python
import pandas as pd
import numpy as np

np.random.seed(30)

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
 
    return alpha
 

def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
 
    return beta
 

def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)
 
    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
 
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator
 
        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
 
        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
 
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)
 
        b = np.divide(b, denominator.reshape((-1, 1)))
 
    return (a, b)
 

def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
 
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])
 
    prev = np.zeros((T - 1, M))
 
    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])
 
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)
 
            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("q0")
        elif s==1:
            result.append("q1")
        elif s==2:
            result.append("q2")
 
    return result
 

    
data = pd.read_csv('./data.csv')
 
V = data['Visible'].values
 
# Transition Probabilities
a = np.ones((3, 3))
a = a / np.sum(a, axis=1)
 
# Emission Probabilities
b = np.ones((3,5))
b = b / np.sum(b, axis=1).reshape((-1, 1))
 
# Equal Probabilities for the initial distribution
initial_distribution = np.array((1.0, 0.0, 0.0))
 
transition, emission = baum_welch(V, a, b, initial_distribution, n_iter=100)
print('Transition')
print(transition)
print()

print('Emssion')
emission = emission / np.sum(emission, axis=1).reshape((-1, 1))
print(emission)
print()

pred = viterbi(V, transition, emission , initial_distribution)

count = 0  
TP = 0  
FP = 0  
   
for i,p in enumerate(pred):
    if p == 'q1':
        FP+=1
        if p == data['Hidden'][i]:
            FP-=1
            TP+=1
    if p == data['Hidden'][i]:
        count+=1
        
print('Accuracy',count/len(data))
print('Precision', TP/(TP+FP))  
   
print(pred)  
```

    Transition
    [[0.         0.5        0.5       ]
     [0.02941173 0.48529414 0.48529414]
     [0.02941173 0.48529414 0.48529414]]
    
    Emssion
    [[1.28376738e-27 2.60099443e-23 1.86212486e-23 3.11269808e-28
      1.00000000e+00]
     [2.64705873e-01 2.94117637e-01 2.05882346e-01 2.35294109e-01
      3.51426358e-08]
     [2.64705873e-01 2.94117637e-01 2.05882346e-01 2.35294109e-01
      3.51426358e-08]]
    
    Accuracy 0.6388888888888888
    Precision 0.6176470588235294
    ['q0', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q1', 'q0']


    /root/anaconda3/envs/test/lib/python3.7/site-packages/ipykernel_launcher.py:71: RuntimeWarning: divide by zero encountered in log
    /root/anaconda3/envs/test/lib/python3.7/site-packages/ipykernel_launcher.py:78: RuntimeWarning: divide by zero encountered in log


위의 결과를 살펴보게 되면 Accuracy는 64%의 결과를 얻었으나, 처음과 마지막을 q0라고 판단하고, 그 외에는 q1으로 판단하게 된다. 이러한 결과는 Initialization을 잘못 하였다고 추측하였다. 즉, Accuracy는 높아도 Model의 Precision은 많이 부족한 상황이라고 판단할 수 있다.

**HMM은 E-M Algorithm이다. 즉, Local Minima or Maxima에 빠질 수 있는 상황이 이다. 따라서 우리는 이러한 것을 해결하기 위하여 기존에 가지고 있는 Data를 가지고 Initialization값을 관측된 Data의 MLE값으로서 변경하여 값을 지정할 수 있다.**

**MLE of Emission Probability**


```python
# Calculate Emission Probability  
q0 = data[data["Hidden"]=="q0"]  
q1 = data[data["Hidden"]=="q1"]  
q2 = data[data["Hidden"]=="q2"]  

q0_mle_list = []
q1_mle_list = []
q2_mle_list = []  


for i in range(5):
    q0_mle_list.append(len(q0[q0['Visible']==i])/len(q0))  

    
for i in range(5):
    q1_mle_list.append(len(q1[q1['Visible']==i])/len(q1))  

    
for i in range(5):
    q2_mle_list.append(len(q2[q2['Visible']==i])/len(q2))  

    
print('q0 MLE Probability')
print(q0_mle_list)
print()  


print('q1 MLE Probability')
print(q1_mle_list)
print()  

print('q2 MLE Probability')
print(q2_mle_list)  
emission_initial = np.stack((q0_mle_list,q1_mle_list,q2_mle_list),axis=0)  
```

    q0 MLE Probability
    [0.0, 0.0, 0.0, 0.0, 1.0]
    
    q1 MLE Probability
    [0.2857142857142857, 0.23809523809523808, 0.19047619047619047, 0.2857142857142857, 0.0]
    
    q2 MLE Probability
    [0.23076923076923078, 0.38461538461538464, 0.23076923076923078, 0.15384615384615385, 0.0]


**MLE of Transimission Probability**


```python
# Calculate Transimission Probability  
transmission_array = np.array(((0,0,0),(0,0,0),(0,0,0)))  
d = ["q0","q1","q2"]  

for i in range(len(data)-1):
    before = data["Hidden"][i]
    after = data["Hidden"][i+1]  
    
    for i,value in enumerate(d):
        for j,value2 in enumerate(d):
            if before == value and after == value2:
                transmission_array[i,j]+=1  

                
transmission_initial = transmission_array / np.sum(transmission_array, axis=1).reshape((-1, 1))  
print('Transmission of Probability')  
print(transmission_initial)  
```

    Transmission of Probability
    [[0.         1.         0.        ]
     [0.04761905 0.85714286 0.0952381 ]
     [0.         0.15384615 0.84615385]]


**Implement Viterbi Algorithm in Hidden Markov Model using Python with MLE Initial Probability**


```python
transition, emission = baum_welch(V, transmission_initial, emission_initial, initial_distribution, n_iter=100)
print('Transition')
print(transition)
print()

print('Emssion')
emission = emission / np.sum(emission, axis=1).reshape((-1, 1))
print(emission)
print()

pred = viterbi(V, transition, emission , initial_distribution) 

count = 0  
TP = 0  
FP = 0  
   
for i,p in enumerate(pred):
    if p == 'q1':
        FP+=1
        if p == data['Hidden'][i]:
            FP-=1
            TP+=1
    if p == data['Hidden'][i]:
        count+=1
        
print('Accuracy',count/len(data))
print('Precision', TP/(TP+FP))  
   
print(pred)
```

    Transition
    [[0.         1.         0.        ]
     [0.06640658 0.32785813 0.60573529]
     [0.         0.48157408 0.51842592]]
    
    Emssion
    [[0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
      1.00000000e+00]
     [5.63565953e-01 4.81671036e-05 7.71107067e-05 4.36308770e-01
      0.00000000e+00]
     [2.71049580e-02 5.27909937e-01 3.69502457e-01 7.54826478e-02
      0.00000000e+00]]
    
    Accuracy 0.6111111111111112
    Precision 0.7058823529411765
    ['q0', 'q1', 'q1', 'q2', 'q2', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q1', 'q1', 'q2', 'q2', 'q1', 'q2', 'q1', 'q2', 'q1', 'q1', 'q2', 'q2', 'q1', 'q2', 'q2', 'q2', 'q1', 'q2', 'q2', 'q1', 'q1', 'q2', 'q2', 'q1', 'q0']


    /root/anaconda3/envs/test/lib/python3.7/site-packages/ipykernel_launcher.py:71: RuntimeWarning: divide by zero encountered in log
    /root/anaconda3/envs/test/lib/python3.7/site-packages/ipykernel_launcher.py:78: RuntimeWarning: divide by zero encountered in log


위의 결과를 살펴보게 되면, Accuracy(64% -> 61%)는 떨어졌으나, 훨씬 더 Precision(62% -> 70%)이 높아진 상황이라고 할 수 있다.
상황에 따라서, 더 좋은 Initialization을 선택하면 될 것이다.

**Dataset의 적고 Precision의 중요도에 따라서 위의 Model에서 Initialization을 어떻게 할지 정하는 것이 더 좋은 방법이라고 생각된다.**

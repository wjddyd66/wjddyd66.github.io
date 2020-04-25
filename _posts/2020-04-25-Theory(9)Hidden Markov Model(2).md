---
layout: post
title:  "Theory9. Hidden Markov Model(2)"
date:   2020-04-25 11:10:20 +0700
categories: [Machine Learning]
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

- 9.5 Hidden Markov Code

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

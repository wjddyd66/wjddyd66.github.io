---
layout: post
title:  "NeuralNetwork (2) Loss Function"
date:   2019-07-26 10:30:00 +0700
categories: [AI]
---

### Loss Function
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
머신러닝에서 Optimazation을 통하여 Loss Function에서 cost(loss)가 최소가 되는 부분을 찾는다.  

이러한 과정을 위해서 Loss Function을 정의하게 되는데  대표적인 방법인 MSE와 음의 로그우도(negative log-likelihood)에 대해서 알아보자.  

### MSE(mean square error)
MSE(mean square error)는 오차의 제곱에 대한 평균을 취한 값으로 통계적 추정의 정확성 에대한 질적인 척도로서 많이 사용된다.  
식은 아래와 같이 나타내게 된다.  
<p>$$f_c(x)=\sum_{i=0}^n  (y_i-\hat{y_i})^2$$</p>
<p>$$y_i: 실제값, \hat{y_i}: 예측값, n: 데이터의 개수$$</p>
MSE가 많이 사용되나 제곱을 하기 때문에 노이즈 데이터에 약한 모습을 보여준다.  
이러한 경우 절대값 손실 함수를 사용하기도 한다. 절대값 손실함수는 아래와 같은 식으로서 표현할 수 있다.  
<p>$$f_c(x)=\sum_{i=0}^n  |(y_i-\hat{y_i})|$$</p>
위의 식은 아래와 같은 코드로 간단히 구현될 수 있다.  
```python
#MSE선언
def MSE(y,y_):
    return 0.5*np.sum((y-y_)*(y-y_))
```
실제 구현한 MSE로서 Loss를 구해보면 다음과 같은 결과를 얻을수 있다.  
```python
#MSE로 Error 구해보기
data = [0,0,1.0,0,0,0,0,0,0,0]
y1 =  np.random.rand(10)
y2 = np.random.rand(10)

error1 = MSE(data,y1)
error2 = MSE(data,y2)

print('Error1: ',error1) #Error1:  2.1434971942835075
print('Error2: ',error2) #Error2:  2.0271222025788003
```
MSE와 절대값 손실함수는 주로 <span style ="color: red">**회귀에서 사용**</span>  

### 음의 로그우도(negative log-likelihood)
음의 로그우도(negative log-likelihood)는 주로 <span style ="color: red">**분류(classification)에서 사용한다.**</span>  

음의 로그우도(negative log-likelihood)를 이해하기 위해서 <span style ="color: red">**정보량, 엔트로피**</span>의 개념을 알고 있어야 한다.  
<br>
<span>**정보량**</span>은 아래와 같은 식으로 표현할 수 있다.  

<p>$$I(x) = log(\frac{1}{p(x)}) $$ </p><br>

<span>$$1 \over p(x) $$ </span>은 사건이 발생할 수 있는 확률이다.  
이러한 값에 log를 취함으로 인하여 <span>**필요한 최소한의 자원**</span>을 나타낸다.  
<br>
<span>**Entropy**</span>는 아래와 같은 식으로 표현할 수 있다.  

<p>$$H_p(X)=\sum_{i=0}^n  p(x_i)log(p(x_i)) $$ </p><br>

Entropy는 <span>**정보량에 대한 기댓값이며 동시에 사건을 표현하기 위해 요구되는 평균 자원이라고 할 수 있다.**</span>으로 정의된다.  
예측이 어려울수록 정보의 양은 더 많아지고 엔트로피는 더 커진다.  
<br>
<span>**Cross Entropy**</span>의식은 아래와 같다.  

<p>$$H(P,Q)= -\sum_{x}  P(x)log(Q(x)) $$ </p><br>

Entropy는 <span>**P는 true label에 대한 분포를, Q는 현재 예측모델의 추정값에 대한 분포**</span>를 의미하게 된다.  

즉, 우리는 true label에 대한 분포와 예측모델의 추정값에 대한 분포를 동일하게 만드는 것을 최종적인 목표로 한다는 것을 알 수 있다.  
다음과 같은 간단한 식으로서 Cross Entropy를 선언할 수 있다.  
```python
#Cross Entropy 선언
def CE(y,y_):
    delta = 1e-7
    return -np.sum(y_*np.log(y+delta))
#y가 0일경우 infinite를 반환하므로 매우 적은 값을 더해주는 과정이 필요
```
실제 구현한 Cross Entropy로서 Loss를 구해보면 다음과 같은 결과를 얻을수 있다.  
```python
error1 = CE(np.array(data),y1)
error2 = CE(np.array(data),y2)

print('Error1: ',error1) #Error1:  6.447238200383332
print('Error2: ',error2) #Error2:  14.50628607586249
```
<br><br>
**크로스엔트로피계산 예시**

범주가 2개인 정답 레이블 [1,0] 인 관측치 x가 있다고 가정하자.  

P는 우리가 가지고 있는 데이터의 분포를 나타낸다.  

Q는 P에 근사하도록 만들고 싶은, 모델이 예측하는 분포를 나타낸다.  

만약 Model의 학습이 잘 되어서 Q가 [1,0] 의 값이 나오게 된다면 Cross Entropy로서 계산하면 아래와 같은 결과가 나오게 된다.  

<p>$$ -P(x)log(Q(x)) = \begin{bmatrix} -1 && 0 \end{bmatrix} \begin{bmatrix} log1 \\ log0 \end{bmatrix} = -(0 +0) = 0$$ </p>

만약 Model의 학습이 잘못 되어서 Q가 [0,1] 의 값이 나오게 된다면 Cross Entropy로서 계산하면 아래와 같은 결과가 나오게 된다.  

<p>$$ -P(x)log(Q(x)) = \begin{bmatrix} -1 && 0 \end{bmatrix} \begin{bmatrix} log0 \\ log1 \end{bmatrix} = -(-\infty +0) = \infty$$ </p>

<hr>
참조:<a href="https://github.com/wjddyd66/Tensorflow/blob/master/Loss%20Function.ipynb">원본코드</a>
참조: <a href="https://ratsgo.github.io/deep%20learning/2017/09/24/loss/">ratsgo 블로그</a> <br>
참조:<a href="http://blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221386519587&categoryNo=52&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=search&userTopListOpen=true&userTopListCount=5&userTopListManageOpen=false&userTopListCurrentPage=1">예비계발자 블로그</a><br>
참조: 밑바닥 부터 시작하는 딥러닝<br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
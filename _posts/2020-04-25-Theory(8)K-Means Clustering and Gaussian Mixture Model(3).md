---
layout: post
title:  "Theory8. K-Means Clustering and Gaussian Mixture Model(3)"
date:   2020-04-25 10:58:20 +0700
categories: [Machine Learning]
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 8. K-Means Clustering and Gaussian Mixture Model(3)
$$\newcommand{\argmin}{\mathop{\mathrm{argmin}}\limits}$$
$$\newcommand{\argmax}{\mathop{\mathrm{argmax}}\limits}$$
Machine Learning의 기초적인 이론부분을 다시 제대로 잡고 싶어서 <a href="https://kaist.edwith.org/machinelearning2__17/joinLectures/9782">문일철 교수님의 인공지능 및 기계학습 개론</a>을 정리한 Post입니다.

- 8.6 EM step for Gaussian Mixture Model
- 8.7 Relation between K-means and GMM
- 8.8 Fundamentals of the EM Algorithm
- 8.9 Derivation of EM Algorithm

### 8.6 EM step for Gaussian Mixture Model
먼저 EM Algorithm을 적용하기 위해서는 Multivariate Gaussian Distribution의 MLE를 알아야 한다.  

해당 과정은 <a href="https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian">StackExchange</a>에 잘 나와있어서 가져왔다.  

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/HandsOn/Theory/30.png)  

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/HandsOn/Theory/31.png)  

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/HandsOn/Theory/32.png)  

위의 식을 활용하여 GMM Model을 EM Algorithm을 활용하여 Maximize하여 보자.  
먼저 위에서 GMM Model을 다음과 같이 선언하였다.  
<p>$$ln(P(X|\pi,\mu,\boldsymbol{\Sigma})) = \sum_{n=1}^{N}ln[\sum_{k=1}^{K} \pi_k N(x|\mu_k,\boldsymbol{\Sigma}_k)]$$</p>

**E-Step**  
E-Step은 해당 Data가 어떤 Cluster에 속하는지 Assign하는 Step이다.  
<p>$$\gamma(z_{nk}) = P(z_k=1|x_n) = \frac{P(z_k=1)P(x|z_k=1)}{\sum_{j=1}^{K}P(z_j=1)P(x|z_j = 1)}= \frac{\pi_k N(x|\mu_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j N(x|\mu_j \boldsymbol{\Sigma}_j)}$$</p>
위의 선언한 식으로서 어떤 Cluster에 속하는지 정하게 된다.  

**처음 Initial상태에서는 <span>$$\mu, \boldsymbol{\Sigma}_j, \pi)$$</span>의 값을 Random하게 주어야 하여 정확한 값을 알 수 없지만 M-Step의 단계의 값을 활용하여 점차적으로 값을 update해나아간다.**  

**M-Step**  
E-step에서 얻은 <span>$$\gamma(z_{nk})$$</span>을 활용하여 E-Step에서 알 수 없었던 값 <span>$$\mu, \boldsymbol{\Sigma}, \pi)$$</span>을 Update하는 과정이다.  
위의 식에 대입하여 <span>$$\hat{\mu_k}, \boldsymbol{\Sigma}_k, \pi_k$$</span>를 찾는 과정은 다음과 같다.

**1. <span>$$\hat{\mu_k}$$</span>**  
위의 Multivariate Gaussian Distribution에서의 <span>$$\hat{\mu_k} =\frac{1}{m}\sum_{i=1}^{m} x_i$$</span>이라고 선언하였다.  

<p>$$\frac{d}{d\mu_k}ln(P(X|\pi,\mu,\boldsymbol{\Sigma})) = \sum_{n=1}^{N} \gamma(z_{nk})\boldsymbol{\Sigma}^{-1}(x_n - \hat{\mu_k})=0$$</p>
<p>$$\rightarrow \hat{\mu_k} = \frac{\sum_{n=1}^{N} \gamma(z_{nk}) x_n}{\sum_{n=1}^{N}\gamma(z_{nk})}$$</p>

즉, Multivariate Gaussian Distribution에서 해당 Guasian Distribution이 될 확률 <span>$$\gamma(z_{nk})$$</span>를 곱해줘서 표시하는 것 이다. 이러한 과정은 M-step에서 구한 <span>$$\gamma(z_{nk})$$</span>값이 있어야 구할 수 있다.  

해당 결과를 살펴보게 되면 **MLE of Multinormial Distribution**인 것을 확인할 수 있다.

**2. <span>$$\boldsymbol{\Sigma}_k$$</span>**  
위의 Multivariate Gaussian Distribution에서의 <span>$$\hat{\Sigma}  = \frac{1}{m} \sum_{i=1}^m \mathbf{(x^{(i)} - \hat \mu) (x^{(i)} -\hat  \mu)}^T 
$$</span>이라고 선언하였다.  

<p>$$\frac{d}{d \boldsymbol{\Sigma}_k}ln(P(X|\pi,\mu,\boldsymbol{\Sigma})) \rightarrow \boldsymbol{\Sigma}_k = \frac{\sum_{n=1}^{N} \gamma(z_{nk}) (x_n - \hat{\mu_k})(x_n - \hat{\mu_k})^T}{\sum_{n=1}^{N} \gamma(z_{nk})}$$</p>
이러한 과정은 M-step에서 구한 <span>$$\gamma(z_{nk})$$</span>값이 있어야 구할 수 있다.  

**3. <span>$$\pi_k$$</span>**  
1,2번과 달리 Lagrange Multiply로서 해결하게 된다.  
<p>$$\frac{d}{d\pi_k}ln(P(X|\pi,\mu,\boldsymbol{\Sigma})) + \lambda(\sum_{k=1}^{K} \pi_k -1) = 0$$</p>
<p>$$ \sum_{n=1}^{N} \frac{N(x|\mu_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j N(x|\mu_j \boldsymbol{\Sigma}_j)} + \lambda = 0$$</p>
<p>$$\therefore \lambda = -N \rightarrow \pi_k = \frac{\sum_{n=1}^{N} \gamma(z_{nk})}{N}$$</p>

**GMM은 위와 같이 EM Algorithm으로서 Parameter가 Update된다. E-step: <span>$$\mu, \boldsymbol{\Sigma}, \pi$$</span>을 활용하여 <span>$$\gamma(z_{nk})$$</span> Expect => M-step: <span>$$\gamma(z_{nk})$$</span>을 활용하여 <span>$$\mu, \boldsymbol{\Sigma}, \pi$$</span>을 Maximize을 반복하는 Algorithm이다. => 따라서 처음 Initialization에 따라서 Local Maximum에 빠질 위험이 있다.**  

### 8.7 Relation between K-means and GMM
위에서 Multivariate Guassian Distribution을 다음과 같이 나타내었다.  
<p>$$N(x|\mu,\boldsymbol{\Sigma}) = \frac{1}{2\pi^{D/2}}\frac{1}{|\boldsymbol{\Sigma}|^{1/2}}exp(-\frac{1}{2}(x-\mu)^T \boldsymbol{\Sigma}^{-1}(x-\mu))$$</p>

위의 식에서 <span>$$\frac{1}{2\pi^{D/2}}\frac{1}{|\boldsymbol{\Sigma}|^{1/2}}$$</span>은 상수로서 제거하게 되고 <span>$$\boldsymbol{\Sigma}^{-1} = \frac{1}{\epsilon}$$</span>으로 치환하면 식을 다음과 같이 쓸 수 있다.  
<p>$$N(x|\mu,\boldsymbol{\Sigma}) \approx exp(-\frac{1}{2\epsilon}(x-\mu)^T(x-\mu))$$</p>

위의 식을 <span>$$\gamma(z_{nk})$$</span>에 대입하게 되면 식을 다음과 같이 변형할 수 있다.
<p>$$\gamma(z_{nk}) = P(z_k=1|x_n) = \frac{\pi_k N(x|\mu_k,\boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K}\pi_j N(x|\mu_j \boldsymbol{\Sigma}_j)} \approx \frac{\pi_kexp(-\frac{1}{2\epsilon}||x-\mu_k)||)^2}{\sum_{j=1}^{K}\pi_jexp(-\frac{1}{2\epsilon}||x-\mu_k)||^2)}$$</p>

위의 식에서 <span>$$\epsilon$$</span>이 0에 가깝게 간다고 생각하게 되면 <span>$$exp(-\frac{1}{2\epsilon}||x-\mu_k)||^2) \rightarrow epx(-\inf) \approx 0$$</span>으로 값이 변경될 것이다.  

또한, <span>$$x-\mu_k$$</span>의 값이 클 수록 더 빨리 수렴하게 될 것이다.  

**최종적으로는 <span>$$x-\mu_k$$</span>차이가 0에 가까운 것만 값이 0이 아니게 되고, 다른 값들은 전부 0이 될 것이다. 이러한 결과는 K-Means Cluster와 같다. 즉, <span>$$\epsilon$$</span>이 0에 가깝게 되면 Soft Clustering -> Hard Clustering으로서 변하게 되면서 K-Means Cluster와 같이 Model을 변경할 수 있다는 의미이다.**

### 8.8 Fundamentals of the EM Algorithm
EM Algorithm을 잘 이해하기 위하여 먼저 Classification과 Clustering에 대하여 조금 더 자세히 알아보자.  
가장 많이 사용되는 분류는 다음과 같다.
- Supervised Learning: Classification
- Unsupervised Learning: Clustering

조금 더 수학적으로 표시하면 다음과 같다.  
- <span>$$[X,Z]$$</span>: Complete set of Variables
- <span>$$X$$</span>: Observed Variables
- <span>$$Z$$</span>: Hidden Variables(Latent Variables)
- <span>$$\theta$$</span>: Parameters for distribution
- <span>$$P(X|\theta)$$</span>
 - Classification: <span>$$P(X|\theta) \rightarrow lnP(X|\theta)$$</span>
 - Clustering: <span>$$P(X|\theta) = \sum_{z}P(X,Z|\theta) \rightarrow lnP(X|\theta) = ln[\sum_{z}P(X,Z|\theta)]$$</span>

위의 수식으로 알 수 있는 사실은 **Unsupervised Learning인 Clustering인 우리가 Observation하지 못하는 Hidden Variable Z(K-means: Center of Cluster, GMM: <span>$$\mu, \boldsymbol{\Sigma}, \pi$$</span>)을 포함하고 있다. 따라서 Supervised Learning과 같이 바로 Optimization이 가능한 것이 아니라 Hidden Variable에 대한 Marginalization후 Optimization하는 과정을 거치게 된다.**  
**우리는 이러한 과정에서 Hidden Variable Z를 실제 Observation할 수 없으므로 정할 수 없고, Model의 결과에 의하여 자동으로 결정되기 때문에 UnSupervised Learning이라고 불리게 된다.**  

**Jensens's Inequality**  
위의 Clustering의 식 <span>$$ln[\sum_{z}P(X,Z|\theta)]$$</span>의 형태는 Log안에 Summation이 존재하기 때문에 계산하기 힘들기 때문에 Summation을 앞으로 뺄 수 있는 방법이 Jensens's Inequality이다. Jensen's Inequality를 살펴보면 다음과 같다.  

<img alt="" src="//upload.wikimedia.org/wikipedia/commons/thumb/c/c7/ConvexFunction.svg/400px-ConvexFunction.svg.png"><br>
사진 출처: <a href="https://en.wikipedia.org/wiki/Jensen%27s_inequality">WIKI</a><br>

- Convex: <span>$$\varphi \left({\frac {\sum a_{i}x_{i}}{\sum a_{i}}}\right)\leq {\frac {\sum a_{i}\varphi (x_{i})}{\sum a_{i}}}\qquad \qquad$$</span>
- Concave: <span>$$\varphi \left({\frac {\sum a_{i}x_{i}}{\sum a_{i}}}\right)\geq {\frac {\sum a_{i}\varphi (x_{i})}{\sum a_{i}}}.\qquad \qquad$$</span>

사진과 식을 살펴보게 되면 이해하기 쉽다.  
실제 사진은 Convex한 어떠한 Function이 있는경우 특정 구간의 Function의 Summation합은 항상 Function을 거치지 않은 Expectation의 값보다 작다는 것이다.  

**Probability Decomposition**  
위의 Jensens's Inequality를 활용하여 수식을 다시 정리하면 다음과 같이 나타낼 수 있다.
<p>$$l(\theta) = lnP(X|\theta) = ln[\sum_{z}P(X,Z|\theta)] = ln[\sum_{z}q(z)\frac{P(X,Z|\theta)}{q(z)}]$$</p>
Jensens's Inequality(Log Function => Concave Function)  
<p>$$ln[\sum_{z}q(z)\frac{P(X,Z|\theta)}{q(z)}] \ge \sum_{z}q(z)ln[\frac{P(X,Z|\theta)}{q(z)}]$$</p>

위의 수식으로 인하여 Log안에 있는 Summation을 밖으로 빼낼 수 있었다.  
**Jensens's Inequality에서 중요한 것은 수식 그대로 Inequality이다는 것 이다. 즉, 우리는 Summation을 Log밖으로 빼서 계산이 편하지만 Equality가 성립하지 않는 다는 것 이다. 하지만, 이 값보다는 항상 실제 Function의 값을 크니깐 Low Boundary로서 생각할 수 있고, 이러한 Low Boundary를 Maximize하는 것은 실제 Function을 Maximize하는데 도움을 줄 수 있다는 것이다.**  

최종적으로 Maximize하려는 Low Boundary의 식을 정리하면 다음과 같다.  
<p>$$\sum_{z}q(z)ln[\frac{P(X,Z|\theta)}{q(z)}] = \sum_{z}q(z)ln[P(X,Z|\theta)]-q(z)ln[{q(z)}]$$</p>
<p>$$=\sum_{z}q(z)ln[P(X,Z|\theta)]+H(q) (\because H(q) = -q(z)ln[{q(z)}])$$</p>

**위의 수식에서 q(z)가 Probability Density Function이면 Entropy수식이 되는 것을 확인할 수 있다.**  
따라서 최종적인 Maximize하고자 하는 식은 다음과 같이 표현할 수 있다.  
<p>$$\sum_{z}q(z)ln[P(X,Z|\theta)]+H(q)$$</p>

### 8.9 Derivation of EM Algorithm
위에서 정의한 식으로서 EM Algorithm을 실제로 Optimization하는 방법에 대해서 알아보기 전에 필요한 다른 Algorithm은 KL Divernce이다.  

**KL Divergence**  
먼저, KLD(Kullback-Leibler Divergence)는 두 확률분포의 차이를 계산하는데 사용하는 함수이다.  

<p>$$KL(p \| q) = H(p,q) - H(p)$$</p>
<p>$$
% <![CDATA[
KL( p \ \| \ q) =
\begin{cases}
 {\displaystyle \sum_{i} p_i \log \frac{p_i}{q_i}} \,\,또는\,\,-{\displaystyle \sum_{i} p_i \log \frac{q_i}{p_i}} \ \ (이산형) &\\[2ex]
 {\displaystyle \int p(x) \log {\frac {p(x)}{q(x)}}dx} \,\,또는\,\,  -{\displaystyle \int p(x) \log {\frac {q(x)}{p(x)}}dx}\ \ (연속형)&\\[2ex]
\end{cases} %]]>
$$</p>
<p>$$H(p) = -plog(p) \text{: Entropy}$$</p>
<p>$$H(p,q) = -plog(q) \text{: Cross Entropy}$$</p>

위의 수식에서 모든 p와 q가 동일한 값을 가지고 있으면, 즉 두 확률분포가 같다면 log안의 값은 항상 1로서 KL(p||q)는 0의 값을 가지게 될 것이다.  

또한 H(p)는 고정된 값으로서 p와 q의 분포가 다르면 다를 수록, H(p,q)은 Uncertainty가 증가하게 되어서 KL(p||q)는 값이 커지는 것을 알 수 있다.  

이미 H(p)에서 일정한 Uncertainty가 있는데 더욱더 Uncertainty가 증가한다고 생각하면 된다.(Jensens's Inequality에서 LowerBounary가 Entropy(H(p))이다.)  

즉 KLD는 다음과 같은 특징을 같게 된다.

**1. <span>$$KL( p|| q) \ge 0$$</span>**  
Jensens's Inequality에 대입하여 생각하면 다음과 같다.  
<p>$$
KL(p\|q)=−∫p(x)\log{\frac{q(x)}{p(x)}}dx≥−\log ∫p(x)\frac{q(x)}{p(x)}dx= -\log ∫q(x)dx = -\log 1 = 0 \\[2ex]
\\ \ \\
\therefore KL(p\|q) \ge 0 (\because H(q) \ge 0)
$$</p>

위의 수식에서 Entropy가 항상 0보다 크다가 잘 이해가 되지 않으시면 <a href="">2. Fundamentals of Machine Learning</a>를 참조하시면 됩니다.

**2. <span>$$KL( p| q) \neq KL(q | p)$$</span>**  
<p>$$
% <![CDATA[
\begin{align}
KL(p\|q) & = H(p,q) - H(p) \\
& \neq H(q, p) - H(q) = KL(q\|p) \\
\end{align}
\\ \ \\
\\ \ \\
 \therefore KL(p\|q) \neq KL(q\|p) %]]>
$$</p>

따라서 위의 수식말고 JSD(Jensen-Shannon Divergence)를 사용하기도 합니다.
<p>$$
JSD(p\|q) = \frac{1}{2}KL(p\|M)  + \frac{1}{2}KL(q\|M) \\
where, M = \frac{1}{2}(p+q)
$$</p>

**Derivation of EM Algorithm**  
이제 Derivation of EM Algorithm에 필요한 공식들을 알았으므로 증명을 해나가면 다음과 같다.  

**E-Step**  
Jensens's Inequality  
<p>$$l(\theta) = lnP(X|\theta) = ln[\sum_{z}q(z)\frac{P(X,Z|\theta)}{q(z)}] \ge \sum_{z}q(z)ln[\frac{P(X,Z|\theta)}{q(z)}] = Q(\theta,q)$$</p>
<p>$$(q = [X,Z])$$</p>

Conditional Probability  
<p>$$\sum_{z}q(z)ln[\frac{P(X,Z|\theta)}{q(z)}] = \sum_{z}q(z)ln[\frac{P(Z|X,\theta)P(X|\theta)}{q(z)}]$$</p>
<p>$$= \sum_{z}q(z)(ln[P(Z|X,\theta)P(X|\theta)+q(z)]) = lnP(X|\theta)+\sum_{z}q(z)ln\frac{P(Z|X,\theta)}{q(z)}$$</p>

위의 식을 Jenson's Inequality에 관련되어 생각해보면 다음과 같이 나타낼 수 있다.
<p>$$lnP(X|\theta) \ge lnP(X|\theta)-\sum_{z}q(z)ln\frac{q(z)}{P(Z|X,\theta)} = L(\theta,q)$$</p>

**즉, <span>$$-\sum_{z}q(z)ln\frac{q(z)}{P(Z|X,\theta)}$$</span>때문에 Jenson's Inequality가 발생하는 것을 알 수 있다. 이 값을 0으로 만드는 것이 Low Boundary를 Maximize하는 과정이 된다.**  

KL Divergence  
<p>$$KL(q(z)||P(Z|X,\theta)) = \sum_{z}q(z)ln\frac{q(z)}{P(Z|X,\theta)}$$</p>
위의 값을 0으로 만들기 위하여 <span>$$q(z) = P(Z|X,\theta)$$</span>로 설정해야 한다는 것을 알았다.

**M-Step**  
**위에서 <span>$$q(z) = P(Z|X,\theta)$$</span>로서 우리는 Latent Variables인 z에 대하여 알 수 있었다.  따라서 원래 Maximize하고자 하는 <span>$$Q(\theta,q), q=[X,Z]$$</span>에서 q의값을 알 수 있으므로 MLE를 통하여 Maximize할 수 있다.**  

이러한 과정을 그림으로서 표현하면 다음과 같다.  
<img src="https://k.kakaocdn.net/dn/bpLphg/btqy0S807LA/FKr4ZfD5tjc5HJ5WB8l2IK/img.png"><br>

Hidden Variable에 대하여 예측(E-Step)하고 그것을 활용하여 Model을 Optimize(M-Step)하는 방법을 반복하는 것을 알 수 있다.  

**중요한 것은 처음 Initialization에 따라서 예측해야 하는 q가 정해지므로 Local Minimum에 빠질 수 있다는 것이다.**

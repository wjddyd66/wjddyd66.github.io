---
layout: post
title:  "Theory7. Bayesian Classifier"
date:   2020-04-25 10:10:20 +0700
categories: [Handson]
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 7. Bayesian Classifier
$$\newcommand{\argmin}{\mathop{\mathrm{argmin}}\limits}$$
$$\newcommand{\argmax}{\mathop{\mathrm{argmax}}\limits}$$
Machine Learning의 기초적인 이론부분을 다시 제대로 잡고 싶어서 <a href="https://kaist.edwith.org/machinelearning2__17/joinLectures/9782">문일철 교수님의 인공지능 및 기계학습 개론</a>을 정리한 Post입니다.

- 7.1 Probability Concepts
- 7.2 Probability Theorems
- 7.3 Interpretation of Bayesian Network
- 7.4 Bayes Ball Algorithm
- 7.5 Factorization of Bayesian networks
- 7.6 Inference Question on Bayesian Network
- 7.7 Variable Elimination
- 7.8 Potential Function and Clique Graph
- 7.9 Simple Example of Belief Propagation

### 7.1 Probability Concepts, 7.2 Probability Theorems
기본적인 확률에 대한 이야기 이다.  
기초적인 부분이며 다루는 범위가 많기 때문에 짧게 다루지 않고 <a href="https://wjddyd66.github.io/categories/#statistics">Statistics</a>에 하나의 Chapter로서 정리해 두었다.  

앞으로 계속 사용하게 될 결과만 정리하면 다음과 같다.  
**Chain Rule or Factorization**  
<p>$$P(a,b,c, ... z) = P(a|b,c,... z)P(b|c, ... z)P(c|...z)...P(z)$$</p>
Joint Probability는 Conditional Probabilty로 인하여 Factorization형태로 나타낼 수 있다.

**Independence**  
<p>$$P(A|B) = P(A)$$</p>
<p>$$P(A,B) = P(A)P(B)$$</p>

즉, B에 관계없이 Conditional Probability는 A와만 상관있다는 상황이다. => Independent하지 않으면 Joint Distribution에서 값을 구하기 복잡해진다.

- Marginal Independence: <span>$$P(X,Y) = P(X)P(Y), P(X) = P(X|Y)$$</span>
- Conditional Independence: <span>$$P(A|B,C) = P(A|C) \rightarrow P(A,B|C) = P(A|B,C)P(B|C) = P(A|C)P(B|C)$$</span> => C가 Given인 상황(Condition)에서는 B에 대해서 Independence하다.

**Marginal Independence Conditional Independence는 앞으로 알아보게 될 Bayesian Network의 핵심이기 때문에 예시를 사용하여 알아보자.**  
<img src="https://k.kakaocdn.net/dn/cd5i9i/btqwdMXNRrU/r4XVZr8KyeI7y1HfnU1kX0/img.png"><br>
참조: <a href="https://deadsquart.tistory.com/27?category=826664">deadsquart 블로그</a><br>

>Commander : Y , OfficerA : X1, OfficerB : X2 라 가정.    
OfficerA는 Commander 가 Go 라고 명령을 내렸는지 모른다고생각해보자. 그래서 OfficerA 가 Go 할지 말지 고민하고 있는데 , 옆의 OfficerB가 GO 하고 있어서 나도 GO를 했다고 해보자.  
이렇게 되면 OfficerB의 행동(정보) 가 OffierA의 행동에 영향을 준것이다. 즉 두사람의 관계가 독립적이지 않다.  <br><br>
이번에는 OfficerA는 Commander 가 Go 라고 명령을 내린것을 들었다고 한다면 (조건) , 옆의 Officer B가 GO 하던 말던 옆사람의 정보(행동)에 관계없이 OfficerA 는 Go 하게 된다.
즉 Y(Latent variable)가 Given 이면(관측이 되었다면) X1, X2 사이에 Conditional independent 를 가정할수 있다.    <br><br>
Marginal independence 는 X1, X2 사이의 관계가 정의되지 않았다.  
Conditional independence 는 Latent variable의 정보만이 X1의 행동에 영향을 준다. X1, X2 사이의 관계를 서로 독립이라 정의하였다.  
>

### 7.3 Interpretation of Bayesian Network

이전 <a href="">NaiveBayes Classifier</a>에서 선언한 식을 다시 살펴보면 다음과 같다.  
<p>$$f^{*} = \argmax_{Y=y} P(Y=y|X=x) = \argmax_{Y=y}P(X=x|Y=y)P(Y=y)$$</p>
<p>$$\approx \argmax_{Y=y}\prod_{i=1}^n P(X_i=x_i|Y=y)P(Y=y)$$</p>

즉, Y가 Given인 상황에서 X에 대한 확률을 최대화 하는 방법이다.  

**Bayesian Network는 이러한 관계를 Graphical Notation으로서 표현한 것이다. 즉, Node와 Link를 활용하여 NaiveBayes Classifier와 같은 Classifier를 Graphical하게 연결한 것 이다.**  

**Syntax**
- A acylick and directed graph: Graph는 Cycle한 구조가 되어서는 안되고 Direct가 있어야 된다.
- A set of nodes => Node는 Random variable이고, Node가 Link로 이어져 있으면 이어진 Parentes에 대한 Conditional Probability를 의미하게 된다.
 - A random variable
 - A conditional distribution given its partents = <span>$$P(X_i|Parents(X_i))$$</span>
- A set of links
 - Direct influence from the parent to the child

위의 Syntax를 그림으로서 살펴보게 되면 다음과 같이 나타낼 수 있다.  
<img src="http://norman3.github.io//prml/images/Figure8.15.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>

위의 그림을 살펴보게 되면 각각의 a,b,c는 Random Variable이며, a,b는 Parent c를 가지고 있다.  

### 7.4 Bayes Ball Algorithm
**참조**: 이번 Part 경우는 <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a>를 많이 참조하여 정리하였습니다.

**Bayes Ball Algorithm이라는 것은 Bayesian Network의 Graph구조에서 Node끼리 Independent가 되기 위하여 관측되어야 할 Node를 정하는 Algorithm이다.**  
Conditional Independence의 조건을 다시한번 생각해보면 다음과 같다.  
<p>$$P(a,b,c) = p(a|b,c)p(b|c)p(c) (\because \text{Factorization}) \rightarrow^{if a \bot b | c} p(a|c)p(b|c)p(c)$$</p>

Conditional Independence를 적용하여 Bayesian Network에서 대표적인 3가지 경우에 대해서 생각하면 다음과 같다.  

**Diverging Connections(Common parent)**  
<img src="http://norman3.github.io//prml/images/Figure8.15.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>
위와 같은 상황일 경우 다음을 만족하지 않는다.  
<p>$$p(a,b,c) = p(a|b,c)p(b|c)p(c) \neq p(a|c)p(b|c)p(c) \rightarrow a \not\bot b | 0$$</p>

**위의 상황에서 만약, C가 Observation되었다고 하면 다음과 같이 변경 될 수 있다.(예시에서 Command가 명령을 내렸을 경우랑 같은 상황이다.)**

<img src="http://norman3.github.io//prml/images/Figure8.16.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>


<p>$$p(a,b|c) = \frac{p(a,b,c)}{p(c)} = p(a|c)p(b|c)$$</p>
<p>$$a\bot b | c$$</p>
위와 같은 상황은 다음과 같이 정리할 수 있다.

- 노드가 두개의 화살표 꼬리 부분에 연결되어 있기 때문에 tail-to-tail 형태라고 한다.
- **노드 c를 관찰하게 되면 a와 b까지의 경로를 차단(block)하게 되어 a와b가 서로 조건부 독립이 된다.**


**Serial Connections(Cascading)**  
<img src="http://norman3.github.io//prml/images/Figure8.17.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>
위와 같은 상황일 경우 다음을 만족하지 않는다.  
<p>$$p(a,b,c) = p(a|b,c)p(b|c)p(c) \neq p(a|c)p(b|c)p(c) \rightarrow a \not\bot b | 0$$</p>

**위의 상황에서 만약, C가 Observation되었다고 하면 다음과 같이 변경 될 수 있다.**

<img src="http://norman3.github.io//prml/images/Figure8.18.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>


<p>$$p(a,b|c) = \frac{p(a,b,c)}{p(c)} = p(a|c)p(b|c)$$</p>
<p>$$a\bot b | c$$</p>
위와 같은 상황은 다음과 같이 정리할 수 있다.

- 하나의 화살표는 화살표의 머리가, 하나의 화살표는 화살표의 꼬리가 오기 때문에 head-to-tail 형태라고 한다.
- **노드 c를 관찰하게 되면 a와 b까지의 경로를 차단(block)하게 되어 a와b가 서로 조건부 독립이 된다.**


**Converging Connections(V-Structure)**  
<img src="http://norman3.github.io//prml/images/Figure8.19.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>
위와 같은 상황일 경우 다음을 만족하지 않는다.  
<p>$$p(a,b,c) = p(a|b,c)p(b|c)p(c) \neq p(a|c)p(b|c)p(c) \rightarrow a \not\bot b | 0$$</p>

**위의 상황에서 만약, C가 Observation되었다고 하면 다음과 같이 변경 될 수 있다.**

<img src="http://norman3.github.io//prml/images/Figure8.20.png" width="300px"><br>
사진 출처: <a href="http://norman3.github.io/prml/docs/chapter08/2.html">norman3 블로그</a><br>


<p>$$p(a,b|c) = \frac{p(a,b,c)}{p(c)} = p(a|c)p(b|c)$$</p>
<p>$$a\bot b | c$$</p>
위와 같은 상황은 다음과 같이 정리할 수 있다.

- 노두가 두개의 화살표 머리 부분에 연결되어 있기 때문에 head-to-head 형태라고 한다.
- **노드 c를 관찰하게 되면 a와 b까지의 경로를 차단(block)하게 되어 a와b가 서로 조건부 독립이 된다.**

**즉, 서로 Conditional Independence하지 않은 2개의 Node에 대하여 특정 Node를 observation하게 되면 Conditional Independence한 형태로 변한다는 것 이다.**  


**Markov Blanket**  
<p>$$A \bot B | Blanket$$</p>
<p>$$Blanket = {parents, children, children's other parents}$$</p>

위의 수식을 설명하게 되면, **A라는 RandomVariable에 대하여 Blanket안의 값을 Observation하게 되면, Blanket바깥의 B Randomvariable에 대하여 Conditional Independent하다는 것 이다.**  

하나의 예제를 생각하면 다음과 같은 Bayesian Network와 A Random Variable을 생각하여 보자.  
<img src="https://k.kakaocdn.net/dn/dKdVlb/btquw3OZt3h/wudOKDLkL1P9svazkqzn01/img.png">

Markov Blanket에 의하여 
1. A Random Variable의 Parents => Cascading/Common parents의 경우 방지
2. A Random Variable의 Children => Cascading의 경우 방지
3. A Random Variable의 Childre's other parents: V-Structure의 경우 방지

Markov Blanket의 결과는 다음과 같이 나타낼 수 있다.  
<img src="https://k.kakaocdn.net/dn/ZPzg0/btquwfI8kFM/UVWr4Sk2YXnLBuvtad1HV0/img.png">
A와 노란원의 RandomVariable과의 관계는 Conditional Independence하다.

### 7.5 Factorization of Bayesian networks
Bayesian Network의 확률은 Conditional Independence하다면 다음과 같이 나타낼 수 있다.  
<p>$$P(X) = \prod_{i}P(X_i|X_{\pi_i})$$</p>
<p>$$X_{\pi_i} = X_i \text{ 's Parent}$$</p>

**위와 같은 상황이면 Bayesian Network를 Factorization하였을 때 Parameter의 개수를 많이 줄일 수 있다는 것 이다.**  
예를 들어 다음과 같다.  
<p>$$P(x_1,x_2,x_3) = p(x_1 | x_2,x_3)p(x_2|x_3)p(x_3) \rightarrow^{if x_1 \bot x_3 | x_2} p(x_1|x_2)p(x_2|x_3)p(x_3)$$</p>
즉, <span>$$p(x_1 | x_2,x_3)$$</span>는 각각의 x가 true, false라면 4가지 상황에 대해서 생각해야 하지만, <span>$$p(x_1 | x_2)$$</span>이 되면 2가지 상황에 대해서만 생각하면 된다는 것 이다.  

실제 Bayesian Network를 다음과 같이 주어졌을 경우 Joint Probability를 생각하여 보자.  
![png](../images/16.png)
<p>$$p(x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9)$$</p>
<p>$$=p(x_1)p(x_2)p(x_3|x_1)p(x_4|x_2)p(x_5|x_2)p(x_6|x_3,x_4)p(x_7|x_6)p(x_8|x_5,x_6)$$</p>

### 7.6 Inference Question on Bayesian Network
**Inference Question이라는 것은 Model이 Training이 된 후에, Model에 Query를 날려서 다른 예측값을 뽑아낼 수 있냐는 것 이다.**  
먼저, 사용하게 될 예제는 다음과 같다.  
![png](../images/17.png)

위의 그림을 살펴보게 되면, Bayesian Network로서 구성되어 있고, P(B), P(E), P(A|B,E), P(J|A), P(M|A)에 관한 확률을 모두 알 수 있다.(Model이 Training된 상황이기 때문이다.)  

**Notation**  
먼저, Inference Question모두에서 사용하는 Notation을 정하고 가면 다음과 같다.  
- <span>$$X=[X_1,X_2,...X_N]$$</span>: All random variable
- <span>$$X_V=[X_{k+1}...X_N]$$</span>: Evidence random variable
- <span>$$X_H=X-X_V=[X_1...X_k]$$</span>: Hidden random variable

위의 예제에 맞춰서 적용하면 각각은 다음과 같다.  
- <span>$$X=[B,E,A,J,M]$$</span>: All random variable
- <span>$$X_V=[B,E]$$</span>: Evidence random variable
- <span>$$X_H=X-X_V=[A,J,M]$$</span>: Hidden random variable

**Inference Question1: Likelihood**  
Gin a set ofe, what is the likelihood of the evidence set?  
<span>$$X_V$$</span>의 Likelihood를 구하여 보자.  
**General한 Formation으로 나타내면 다음과 같이 나타낼 수 있다.**  
<p>$$P(X_V) = \sum_{X_H}P(X_H,X_V) = \sum_{X_1}...\sum_{X_k}P(X_1...X_k,X_V)$$</p>
그렇다면 어떻게 계산을 할 수 있는가에 대해 살펴보게 되면, <span>$$P(X_H,X_V)$$</span>는 Full Joint 형식으로 변하게 된다.  
따라서 위의 General Formation을 그림의 Problem에 적용하면 다음과 같이 식을 변형할 수 있다는 것 이다.  
<p>$$P(X_V) = \sum_{X_H}P(X_H,X_V) = \sum_{A}\sum_{J}\sum_{M}P(B)P(E)P(A|B,E)P(J|A)P(M|A)$$</p>
위의 각각의 확률을 구해져 있으므로, 식을 계산할 수 있는 형태로서 변하게 된다.  

**Inference Question 2: Conditional Probability**  
Given a set of evidence, what is the conditional probability of interested hidden variables?  
먼저 Hidden Variable을 다음과 같이 2개의 Variable로서 분류하여 보자.  
<span>$$X_H = [Y,Z]$$</span>
 - Y: Interested hidden variables
 - Z: Un interested hidden variables

<span>$$P(Y|X_V)$$</span>: Evidence가 주어진 경우, Interest한 Variable의 확률을 General한 Formation으로 나타내면 다음과 같다.  
<p>$$P(Y|X_V) = \sum_{z}P(Y,Z=z|X_V)$$</p>
<p>$$\sum_{z} \frac{P(Y,Z,X_V)}{P(X_V)} = \sum_{z} \frac{P(Y,Z,X_V)}{\sum_{y,z}P(Y=y,Z=z,x_v)}$$</p>
위의 식을 자세히 살펴보게 되면 <span>$$P(X_V)$$</span>는 Inference Question1에서 구한 값이다. 즉, Full Joint에서 Uninterested Hidden Variable에 대하여 Marginalization하면 Interested한 Variable의 확률을 구할 수 있다는 것 이다.  
따라서 위의 General Formation을 그림의 Problem에 적용하면 다음과 같이 식을 변형할 수 있다. (Y=J)로서 지정한다.  
<p>$$P(Y|X_V) = \frac{\sum_{A}\sum_{M}P(B)P(E)P(A|B,E)P(J|A)P(M|A)}{\sum_{A}\sum_{J}\sum_{M}P(B)P(E)P(A|B,E)P(J|A)P(M|A)}$$</p>

**Inference Question 3: Most Probable Assignment**  
위의 Bayesian Network의 결과값을 보면 다음과 같이 생각할 수 있다.  
- B,E가 주어지면 A에 관하여 **Prediction** 가능하다. => B,E의 상태에 따라서 A의 확률을 결정할 수 있다.
- A가 주어지면 J,M의 상태에 대하여 **Diagnosis**가 가능하다. => A의 상태에 따라서 J,M의 상태를 확률로 결정할 수 있다.

### 7.7 Variable Elimination
**Marginalization and Elimination**  
Marginalization을 통하여 확률의 곱셈에 대하여 Summation을 실시하게 된다. => 즉, Summation에 관계없는 확률은 Summation안에서 Elimination이 가능하다는 것 이다.  
위의 Inference Question 1을 예시로 들면 다음과 같다.  
<p>$$P(X_V) = \sum_{X_H}P(X_H,X_V) = \sum_{A}\sum_{J}\sum_{M}P(B)P(E)P(A|B,E)P(J|A)P(M|A)$$</p>
<p>$$= P(B)P(E)\sum_{A}\sum_{J}\sum_{M}P(A|B,E)P(J|A)P(M|A)$$</p>
위로서 변경하게 되어서 Computing을 많이 줄일 수 있다.

**Variable Elimination**  
위의 Marginalization and Elimination을 좀더 Generalization한 형태로서 표현해보자.  
구하고자 하는 확률은 <span>$$P(E,J,M)$$</span>이라고 하여보자. 이식을 Full Joint로서 표현하면 다음과 같이 나타낼 수 있다.  
<p>$$P(E,J,M) = \sum_{A}\sum_{B}P(B)P(E)P(A|B,E)P(J|A)P(M|A)$$</p>
위의 식을 Marginalization and Elimination을 통하여 나타내면 다음과 같이 변형시킬 수 있다.
<p>$$P(E,J,M) = P(E)\sum_{B}P(B)\sum_{A}P(A|B,E)P(J|A)P(M|A)$$</p>
위의 수식에서 **<span>$$f_A(b) = P(B|A)$$</span>** 의 형태인 Function으로 정의하면 위의 식을 다음과 같이 정의할 수 있다.  
<p>$$P(E,J,M) = f_E(e)\sum_{B}f_{B}(b)\sum_{A}f_A(a,b,e)f_J(a)f_M(a)$$</p>
위의 수식은 Inference Question on Bayesian Network에서 사용한 Example에 적용하면 다음과 같이 적용할 수 있다.  

![png](../images/18.png)

<p>$$P(E,J,M) = f_E(e)\sum_{B}f_{B}(b)\sum_{A}f_A(a,b,e)f_{JM}(a)$$</p>

![png](../images/19.png)
<p>$$P(E,J,M) = f_E(e)\sum_{B}f_{B}(b)\sum_{A}f_{AJM}(a,b,e)$$</p>

![png](../images/20.png)
<p>$$P(E,J,M) = f_E(e)\sum_{B}f_{B}(b)f_{\bar{A}JM}(a,b,e)$$</p>

위와 같은 형식으로 계속하여 계산할 수 있다.

### 7.8 Potential Function and Clique Graph
**Potential function**: A function which is not a probability function yet, but once normalized it can be probability distribution function  
- Potential function on nodes: <span>$$\psi(a,b),\psi(b,c),\psi(c,d)$$</span> => Clique
- Potential function on links: <span>$$\phi(b), \phi(c)$$</span> => Separateor

![png](../images/21.png)

**Potential Function이라는 것은 PDF로서 표현하기 위한 방법이다. Bayesian Network를 Clique(Fully Conntected)와 Separator로서 표현한다.**  

위의 그림을 Bayesian Network의 특성에 따라서 Full Joint로서 표현하면 다음과 같이 나타낼 수 있다.  
<p>$$P(A,B,C,D) = P(A|B)P(B|C)P(C|D)P(D)$$</p>

위의 식을 각각의 Clique와 Separator로서 표현하면 다음과 같이 표현할 수 있다.  

**Notation 1(Conditional Probability)**  
<p>$$P(A,B,C,D) = \frac{\prod_N \psi(N)}{\prod_L \phi(L)} = \frac{\psi(a,b)\psi(b,c)\psi(c,d)}{\phi(b)\phi(c)}$$</p>
<p>$$\because \psi(a,b) = P(A|B), \psi(b,c) = P(B|C)\psi(c,d) = P(C|D)P(D)$$</p>
<p>$$\because \phi(b)=1, \phi(c) =1$$</p>

**Notation 2(Joint Probability)**  
<p>$$P(A,B,C,D) = \frac{\prod_N \psi(N)}{\prod_L \phi(L)} = \frac{\psi^{*}(a,b)\psi^{*}(b,c)\psi^{*}(c,d)}{\phi^{*}(b)\phi^{*}(c)}$$</p>
<p>$$\because \psi^{*}(a,b) = P(A,B), \psi^{*}(b,c) = P(B,C)\psi^{*}(c,d) = P(C,D)$$</p>
<p>$$\because \phi^{*}(b)=P(B), \phi^{*}(c) =P(C)$$</p>

**Bayesian Network의 Full Joint Factorization을 Clique, Separator로서 Clique-Conditional, Joint / Separator-1,Individual Probability 로서 나타낼 수 있다.**

### 7.9 Simple Example of Belief Propagation
**Belief Propagation이라는 것은 Bayesian Network 특성상 하나의 Random Variable이 Evidence Variable로 변함으로 인하여 연관있는 Probability가 변한다는 의미이다.**  

먼저 위의 상황에서 **Joint Probability의 예시로 살펴보자.**  

**Joint Probability**  
<p>$$P(A,B) \rightarrow P(A=1,B): (\text{A: Observation -> Evidence})$$</p>
<p>$$\psi(A,B) \rightarrow \psi^{*}(A,B) (\because \psi(A,B) = P(A,B))$$</p>

위와 같이 Notation을 정의하면 P(B)에 대하여 각각의 Margianlization을 다음과 같이 나타낼 수 있다.  
<p>$$P(B) = \sum_{A} \psi(A,B)$$</p>
<p>$$P(B) = \sum_{C} \psi(B,C)$$</p>
<p>$$P(B) = \phi(B)$$</p>
<p>$$\therefore P(B) = \sum_{A} \psi(A,B) = \sum_{C} \psi(B,C)= \phi(B)$$</p>

위에서 <span>$$P(A,B) \rightarrow P(A=1,B)$$</span>로서 A가 Random -> Evidence Variable로 변형하였다. 이에 따라서 <span>$$\sum_{A} \psi(A,B)$$</span>값은 변하게 될 것이고, 등호 관계에 있는 모든 값들은 변하게 될 것 이다.  

이러한 과정을 **위의 그림에 대입하면 제일 앞의 Node가 변함에 의하여 모든 Node의 값이 변하게 되는 Belief Propagation이라고 불리게 된다.**  

<p>$$\psi(A,B) \rightarrow \psi^{*}(A,B)$$</p>
<p>$$\phi^{*}(B) = \sum_{A} \psi^{*}(A,B)$$</p>
<p>$$\psi^{*}(B,C) = \psi(B,C)\frac{\phi^{*}(B)}{\phi(B)}$$</p>
<p>$$\because \sum_{C}\psi^{*}(B,C) = \sum_{C} \psi(B,C)\frac{\phi^{*}(B)}{\phi(B)} = \frac{\phi^{*}(B)}{\phi(B)}\sum_{C} \psi(B,C) = \frac{\phi^{*}(B)}{\phi(B)}\phi(B) $$</p>

**Conditional Probability**  
실제 얻을 수 있는 Data는 Joint Probability가 아닌 Joint Probability이다.  
따라서, Bayesian Network에 Conditional Probability를 활용하여 Belief Propagation을 실시하여 보자.  
먼저, Bayesian Network를 다시 다음과 같다고 정의하여 보자.
![png](../images/22.png)
위와 같은 Bayesian Network를 Clique Graph로서 변형하면 다음과 같이 나타낼 수 있다.
![png](../images/23.png)
- <span>$$\psi(a,b) = P(a|b), \psi(b,c) = P(b|c)P(c)$$</span>
- <span>$$\phi(b) = 1$$</span>

위와 같은 경우에 다음과 같은 문제 2개를 생각해 보자.  
**Example 1. P(b)=?**  
<p>$$\phi^{*}(b) = \sum_{a}\psi(a,b) = 1$$</p>
<p>$$(\because \sum_{a}\psi(a,b) = \frac{1}{P(b)}\sum_aP(a,b) = \frac{1}{P(b)}P(b))$$</p>
<p>$$\psi^{*}(b,c) = \psi(b,c)\frac{\phi^{*}(b)}{\phi(b)} = P(b|c)P(c) = P(b,c)$$</p>
<p>$$\phi^{**}(b) = \sum_{c}\psi(b,c) = \sum_{c} P(b,c) = P(b)$$</p>
<p>$$\psi^{*}(a,b) = \psi(a,b)\frac{\phi^{**}(b)}{\phi^{*}(b)} = P(a|b)P(b) = P(a,b)$$</p>
<p>$$\phi^{***}(b) = \sum_{a}\psi^{*}(a,b) = \sum_{a} P(a,b) = P(b)$$</p>

위의 과정을 살펴보면 다음과 같은 결과를 얻을 수 있다.
- Conditional Probability -> Joint Probability
 - <span>$$\psi(a,b) = P(a|b) \rightarrow \psi^{*}(a,b) = P(a,b)$$</span>
- 1 -> Indiviaul Probability
 - <span>$$\phi(b) \rightarrow \phi^{***}(b) = \phi^{**}(b) = P(b)$$</span> => 값이 변하지 않는다. Local consitency를 Maintain하고 있다.

**Example 2. P(b|a=1,c=1)=?**  
a,c가 Random Variable => Observation이 되었다고 생각하여 보자.  
<p>$$\phi^{*}(b) = \sum_{a}\psi(a,b)\delta(a=1) = P(a=1|b)$$</p>
<p>$$\psi^{*}(b,c) = \psi(b,c)\frac{\phi^{*}(b)}{\phi(b)} = P(b|c=1)P(c=1)P(a=1|b)$$</p>
<p>$$\phi^{**}(b) = \sum_{c}\psi(b,c)\delta(c=1) = P(b|c=1)P(c=1)P(a=1|b)$$</p>
<p>$$\psi^{*}(a,b) = \psi(a,b)\frac{\phi^{**}(b)}{\phi^{*}(b)} =\frac{P(b|c=1)P(c=1)P(a=1|b)}{P(a=1|b)}P(a=1|b) = P(b|c=1)P(c=1)P(a=1|b)$$</p>
<p>$$\phi^{***}(b) = \sum_{a}\psi^{*}(a,b)\delta(a=1) = P(b|c=1)P(c=1)P(a=1|b)$$</p>
<p>$$P(b|c=1)P(c=1)P(a=1|b) = \frac{P(b,c=1)}{P(c=1)}P(c=1)\frac{P(b,a=1)}{P(b)} = \frac{P(b,a=1)P(b,c=1)}{P(b)} = P(b|a=1,c=1)$$</p>

위의 과정을 살펴보면 다음과 같은 결과를 얻을 수 있다.
- <span>$$\psi()$$</span>
 - <span>$$\psi(a,b) = P(a|b) \rightarrow \psi^{*}(a,b) = P(b|c=1)P(c=1)P(a=1|b)$$</span>
 - <span>$$\psi(b,c) = P(b|c)P(c) \rightarrow \psi^{*}(b,c) = P(b|c=1)P(c=1)P(a=1|b)$$</span>
- <span>$$\phi()$$</span>
 - <span>$$\phi(b) \rightarrow \phi^{***}(b) = \phi^{**}(b) = P(b|c=1)P(c=1)P(a=1|b)$$</span> => 값이 변하지 않는다. Local consitency를 Maintain하고 있다.

**즉, 위의 Clique Graph에서 (A,B)---(B)---(B,C)중 A,C과 관측되었으므로 => (A=Evidence , B)=(B#)---(B#)---(B#)=(B,C=Evidence), B는 양옆의 Evidence Variable에 의하여 Update되면서, (A,B) = (B) = (A,C)의 관계로 변하게 된다.**

**Example 3. P(b|a=1)=?**  
a,c가 Random Variable => Observation이 되었다고 생각하여 보자.  
<p>$$\phi^{*}(b) = \sum_{a}\psi(a,b)\delta(a=1) = P(a=1|b)$$</p>
<p>$$\psi^{*}(b,c) = \psi(b,c)\frac{\phi^{*}(b)}{\phi(b)} = P(b|c)P(c)P(a=1|b) = P(b,c)P(a=1|b)$$</p>
<p>$$\phi^{**}(b) = \sum_{c}\psi(b,c) = \sum_{c}P(b,c)P(a=1|b) = P(c)P(a=1|b)$$</p>
<p>$$\psi^{*}(a,b) = \psi(a,b)\frac{\phi^{**}(b)}{\phi^{*}(b)} =\frac{P(c)P(a=1|b)}{P(a=1|b)}P(a=1|b) = P(c)P(a=1|b)$$</p>
<p>$$\phi^{***}(b) = \sum_{a}\psi^{*}(a,b)\delta(a=1) = P(c)P(a=1|b)$$</p>

위의 과정을 살펴보면 다음과 같은 결과를 얻을 수 있다.
- <span>$$\psi()$$</span>
 - <span>$$\psi(a,b) = P(a|b) \rightarrow \psi^{*}(a,b) = P(c)P(a=1|b)$$</span>
 - <span>$$\psi(b,c) = P(b|c)P(c) \rightarrow \psi^{*}(b,c) = P(b,c)P(a=1|b)$$</span>
- <span>$$\phi()$$</span>
 - <span>$$\phi(b) \rightarrow \phi^{***}(b) = \phi^{**}(b) = P(c)P(a=1|b)$$</span> => 값이 변하지 않는다. Local consitency를 Maintain하고 있다.

**즉, 위의 Clique Graph에서 (A,B)---(B)---(B,C)중 A가 관측되었으므로 => (A=Evidence , B)=(B#)---(B#)---(B#)!=(B,C), B는 A가 Evidence Variable에 의하여 Update되면서, (A,B) = (B) != (A,C)의 관계로 변하게 된다.**  

=> 아직 C는 Random Variable이므로 C에 따라서 (A,C)의 값은 등호로 변할 수 있다. (P(B,C)를 B에 대하여 Margianlization을 하여야 한다.)

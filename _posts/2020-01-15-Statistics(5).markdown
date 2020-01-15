---
layout: post
title:  "Statistics(5)-Moment Generating Function"
date:   2020-01-20 09:40:20 +0700
categories: [statistics]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
이번 POST는 <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업 내용</a>을 정리한 것 입니다.  
문제나 자세한 내용은 <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a>를 참조하였습니다.  

### 7. 적률생성함수(Moment Generating Function)
**적률**  
확률변수 X의 원점에 대한 r차 적률  
확률변수 <span>$$X^r$$</span>의 기댓값 <span>$$E(X^r)$$</span>를 확률변수 X의 원점에 대한 r차 적률이라 하고 이를 <span>$$\mu r^{'} = X^r$$</span>  
<p>$$\mu r^{'} = E(X^r) =
\begin{cases}
\sum_{x}x^rf(x), & \mbox{이산형} \\
\int_{-\infty}^{\infty}x^rf(x)\,dx, & \mbox{연속형}
\end{cases}
$$</p>

**확률변수의 적률은 분포의 특징을 설명해주는 주요한 역할을 한다. 평균, 분산, 왜도, 첨도는 모두 적률의 함수이다.**  
<span>$$\mu r = E[(X-\mu)^r]$$</span>: 평균 <span>$$\mu$$</span>에 대한 <span>$$r$$</span>차 중심적률  

**적률 생성 함수 <span>$$M_x (t)$$</span>**  
확률변수 X의 적률생성함수  <span>$$M_x (t)=E(e^{tx})$$</span>로 정의한다.  
<p>$$
M_x (t) = 
\begin{cases}
\sum_{x}e^{tx} f(x), & \mbox{이산형} \\
\int_{-\infty}^{\infty}e^{tx} f(x)\,dx, & \mbox{연속형}
\end{cases}
$$</p>
적률생헝함수 구하기  
<p>$$\frac{\partial^r}{\partial t^r}M_{x}(t)|_{t=0} = M_{x}^{(r)}(0) = E(x^r) = \mu r^{'}$$</p>

위에식에서 적률생성 함수를 변형하게 되면 다음과 같다.  
<p>$$\frac{\partial}{\partial t} M_{x}(t)|_{t=0} = \frac{\partial}{\partial t} \int_{-\infty}^{\infty} xe^{tx}f(x)\, dx |_{t=0} = \int_{-\infty}^{\infty} xf(x)\, dx = E(X) = \mu 1^{'}$$</p>

<p>$$\frac{\partial}{\partial t^2}M_{x}(t)|_{t=0} = \frac{\partial}{\partial t^2} \int_{-\infty}^{\infty} x^2e^{tx}f(x)\, dx |_{t=0} = \int_{-\infty}^{\infty} x^2f(x)\, dx = E(X^2) = \mu 2^{'}$$</p>
<p>$$\therefore \frac{\partial^r}{\partial t^r}M_{x}(t)|_{t=0} = M_{x}^{(r)}(0) = E(x^r) = \mu r^{'}$$</p>

ex) 확률변수 X가 B(n,p)를 따를 때 X의 적률생성함수를 구하고 이를 이용하여 <span>$$\mu = np, \sigma^2 = npq$$</span>임을 증명하여라.  
<p>$$B(n,p) = {}_{n}\mathrm{C}_{x} p^x q^{n-x}$$</p>
<p>$$B(n,p) = \sum_{x=0}^{n} {}_{n}\mathrm{C}_{x} p^x q^{n-x} = (p+q)^n$$</p>
<p>$$M_x(t) = \sum_{x} e^{tx}f(x) = \sum_{x=0}^{n} e^{tx} {}_{n}\mathrm{C}_{x} p^x q^{n-x}$$</p>
<p>$$= \sum_{x=0}^{n} {}_{n}\mathrm{C}_{x} (e^t p)^x q^{n-x} = (pe^t+q)^n$$</p>

<p>$$\frac{\partial}{\partial t} M_{x}(t)|_{t=0} = n(pe^t + q)^{n-1}pe^t|_{t=0} = n(p+1)^{n-1}p = np = E(X)$$</p>

<p>$$\frac{\partial}{\partial t^2} M_{x}(t)|_{t=0} = n(n-1)(pe^t + q)^{n-2}pe^tpe^t + n(pe^t+q)^{n-1}pe^t|_{t=0} = n(n-1)p^2+np = np(np+1-p)=np(np+q) = E(X^2)$$</p>
<p>$$V(X) = E(X^2) - (E(X))^2 = np(np+q)-np^2 = npq$$</p>

**적률함수의 특징**  
확률변수 X와 Y가 같은 적률생성함수를 가지면 즉, 모든 t에 대하여 <span>$$M_{x}(t) = M_{Y}(t)$$</span>이면 두 확률변수(x,y)는 같은 확률분포(f(x),g(y))를 가진다.  

먼저 각각의 적률생성 함수를 다음과 같이 정의하자.  
<p>$$M_{x}(t) = \int_{-\infty}^{\infty} e^{tx}f(x)\, dx$$</p>
<p>$$M_{y}(t) = \int_{-\infty}^{\infty} e^{ty}g(y)\, dy$$</p>
위에서 각각의 확률변수를 다음과 같이 나타내자.  
<p>$$x \in R_{x}\text{ , } y \in R_{y}\text{ , } R_{x} \cup R_{y} = A \text{ , } a \in A$$</p>
위에서 정리한 a를 활용하여 각각의 적률생성 함수를 변형하면 다음과 같다.  
<p>$$M_{x}(t) = \int_{-\infty}^{\infty} e^{ta}f(a)\, da$$</p>
<p>$$M_{y}(t) = \int_{-\infty}^{\infty} e^{ta}g(a)\, da$$</p>
위의 식에서 먼저 가정을 <span>$$M_{x}(t) = M_{Y}(t) \rightarrow f(x) = g(y)$$</span>로서 두었기 때문에 대입한다.  
<p>$$M_{x}(t) = M_{y}(t) \rightarrow M_{x}(t) - M_{y}(t) = 0 = \int_{-\infty}^{\infty} e^{ta}(f(a) - g(a))\, da$$</p>
<p>$$\therefore f(a) = g(a) \rightarrow f(x) = g(y) \because e^{ta} > 0$$</p>

<hr>
참조: <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업</a><br>
참조: <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


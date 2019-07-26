---
layout: post
title:  "NeuralNetwork (4) Backpropagation"
date:   2019-07-26 11:30:00 +0700
categories: [AI]
---

### Backpropagation
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

Backpropagation을 알기전에 Chain Rule이라는 것을 먼저 알아야 한다.  
Chain Rule은 합성함수의 미분법이다.  
n변수 함수 <span>$$f(x_1,x_2,x_3,...,x_n)$$</span>에 대해  
<span>$$x_k = g_k(t_1,t_2,t_3,...,t_m) (k=1,2,3,...,n)$$</span>이면  
<span>$$\frac{\partial f}{\partial t_i}=\frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t_i}+\frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t_i}+...+\frac{\partial f}{\partial x_n}\frac{\partial x_n}{\partial t_i} (i=1,2,3,...,m)$$</span>이다.  
Chain Rule에대한 자세한 내용:<a href="http://blog.naver.com/PostView.nhn?blogId=mindo1103&logNo=90103548178">Nenyaffle 블로그</a>  

아래와 그림과 같이 간단한 하나의 신경을 생각해 보자.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/15.PNG" height="200" width="600" />
</div>

위와 같은 신경에 대한 Cost Function을 MSE를 사용하면 Cost Function은 아래와 같은 식으로 표현할 수 있다.  
<p>$$C=\frac{1}{2}(y-y\prime)^{2}$$</p>

Weight를 Update하기 위하여 Gradient Decent를 사용하게 되면 W2에 대한 식은 아래와 같은 식으로 표현할 수 있다.  
<p>$$W_2(Update)=W_2-\alpha\frac{\partial C}{\partial W_2}$$</p>

예측한 값인 <span>$$y\prime$$ </span>은 결국 0 혹은 1로써 표현되는 상수이므로 아래와 같은 식을 유도할 수 있다.  
<p>$$\frac{\partial C}{\partial W_2} = $$</p>
<p>$$(y-y\prime) \frac{\partial y}{\partial W_2} =$$</p> 
<p>$$(y-y\prime) \frac{\partial}{\partial W_2}g(W_2h)$$</p>
(g(x): Sigmoid Function)  

Sigmoid 함수를 미분하게 되면 <span>$$g\prime(x) = g(x)(1-g(x))$$ </span>이다.  
또한 <span>$$g(x) = y(1-y)$$ </span>이다.(Sigmoid는 0 또는 1의 값을 가지는 함수 이므로)  
<br>
위의 식과 Chain Rule을 사용하게 되면 아래와 같은 식을 얻을 수 있다.  
<p>$$\frac{\partial C}{\partial W_2} = $$</p>
<p>$$(y-y\prime)g(W_2h)(1-g(W_2h)) \frac{\partial W_2h}{\partial W_2} = $$</p>
<p>$$(y-y\prime)y(1-y) \frac{\partial W_2h}{\partial W_2} = $$</p>
<p>$$(y-y\prime)y(1-y)h$$</p>

Weight2를 Update하였으므로 W1에 대한 식을 위와같은 과정을 거치게되면 아래와 같은 식으로 표현할 수 있다.  
<p>$$\frac{\partial C}{\partial W_1} = $$</p>
<p>$$(y-y\prime) \frac{\partial y}{\partial W_1} = $$</p>
<p>$$(y-y\prime) \frac{\partial}{\partial W_1} = $$</p>
<p>$$(y-y\prime)g(w_2h)(1-g(w_2h)) \frac{\partial W_2h}{\partial W_1} = $$</p>
<p>$$(y-y\prime)y(1-y) \frac{\partial W_2h}{\partial W_1} = $$</p>
<p>$$(y-y\prime)y(1-y)W_2 \frac{\partial h}{\partial W_1} = $$</p>
<p>$$(y-y\prime)y(1-y)W_2 \frac{\partial}{\partial W_1}g(W_1x) = $$</p>
<p>$$(y-y\prime)y(1-y)W_2h(1-h) \frac{\partial W_1x}{\partial W_1} = $$</p>
<p>$$(y-y\prime)y(1-y)W_2h(1-h)x = $$</p>
<p>$$(y-y\prime)y(1-y)h(1-h)W_2x$$</p>

최종적인 식 2개를 비교하게 되면  
<p>$$(y-y\prime)y(1-y)h$$</p>
<p>$$(y-y\prime)y(1-y)h(1-h)W_2x$$</p>
으로서 겹치는 부분 <span>$$(y-y\prime)y(1-y)h$$</span>가 겹치게 된다.  
또한
<p>$$(y-y\prime)y(1-y)h = \delta_y$$</p>
<p>$$(y-y\prime)y(1-y)h(1-h)W_2 = \delta_h$$</p>라 치환을 하게 되면 <span>$$\delta_{h} = \delta_{y}g\prime (x)W_2$$</span>로서 표현할 수 있다.  

위의 결과를 얻어서 아래와 같이 Hidden Layer가 3개인 층인 망을 아래와 같이 생각해보자.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/16.PNG" height="200" width="600" />
</div>
위와 같은 그림일때 위에서 증명한 수식을 사용하게 되면 아래와 같은 식을 알 수 있다.  

<p>$$\delta3 = \delta yg\prime(x) W_4$$</p>
<p>$$\delta2 = \delta 3g\prime(x) W_3$$</p>
<p>$$\delta1 = \delta 2g\prime(x) W_2$$</p>
위와 같이 처음 <span>$$\delta y$$ </span>를 알게되면 나머지의 값을 쉽게 구할 수 있다.  
이렇게 Weight의 Update는 BackPropagation으로 진행되면서 Update가 된다.  
<hr>
참조: <a href="https://www.youtube.com/watch?v=fhrORKjjU7w&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=25">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="http://blog.naver.com/PostView.nhn?blogId=mindo1103&logNo=90103548178">Nenyaffle 블로그</a><br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
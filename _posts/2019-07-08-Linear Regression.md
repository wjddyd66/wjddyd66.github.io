---
layout: post
title:  "Linear Regression"
date:   2019-07-08 10:00:00 +0700
categories: [AI]
---

### Linear Regression
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

<span style ="color: red">'선형적이다' </span>라는 표현은 영어로 linear 하다 라고 말한다. linear란 line(선)의 형용사 형태입니다. 이 말에서 유추할 수 있듯이, 선형적이란 것은 어떤 성질이 변하는데 그 변수가 1차원적이다, 즉 어떤 신호에 기울기만 곱한 형태와 같다.  
<div><img src="http://www.rfdh.com/bas_rf/begin/images/linear1.gif" height="200" width="600" /></div>
그림출처:<a href="http://www.rfdh.com/bas_rf/begin/linear.htm">www.rfdh.com </a><br>
간단한 수식으로는 y= bx + a로서 표현할 수 있다.  
여기서 우리가 중점적으로 봐야할 것은 <span style ="color: red">**weight: b, bias: a**</span>의 상수 2개이다.  
이 상수 2개를 찾아낼 수 있으면 우리는 앞으로 Input이 들어올 경우 Output을 구할 수 있다.  
<a href="https://wjddyd66.github.io/r/2019/06/17/Regression.html">회귀분석 자세한 내용</a>
### Linear Regression 예시
아래 표와 같은 DataSet이 있다고 가정하자.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>집넓이</td><td>집값(만원)</td>
	</tr>
	<tr>
		<td>10</td><td>3000</td>
	</tr>
		<tr>
		<td>15</td><td>4000</td>
	</tr>
		<tr>
		<td>16</td><td>12000</td>
	</tr>
			<tr>
		<td>...</td><td>...</td>
	</tr>

	<tr>
		<td>60</td><td>30000</td>
	</tr>
	</tbody>
</table>
<br>
위의 집 넓이를 X, 집값을 Y라 하였을때 다음과 같은 식을 얻을 수 있다.  

$$y= w_1 X+w_2 b$$  
위의 식은 아래와 같은 그래프로서 표현할 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/1.png" height="200" width="600" /></div>
실제값과 예측값의 차이는 파란색선의 길이의 합이다.  
실제값과 예측값의 차이는 Cost라고 불리게 되고 이러한 Cost에 대한 식은 아래 식으로 나타낼 수 있다.  

$$f_c(x)=\sum_{i=0}^n  (y_i-\hat{y_i})^2$$  
Cost Function은 아래와 같은 그림으로 나타낼 수 있다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/2.PNG" height="250" width="600" /></div>
Cost가 0이 될 확률을 매우 낮지만 <span style ="color: red">**0에 가까울 때(미분했을때의 기울기가 0인 점)**</span>의 값을 구하는 것이 Cost를 최소로 할 수 있다.  
간단하게 구할 수 있다고 생각하지만 Weight가 많아지게 되면 수식이 복잡하게 되므로 Gradient Decent를 사용하게 된다.  

### Gradient Decent
Gradient Decent는 Cost Function을 W에 애해 편미분하면 현재 W위치에서의 접선의 기울기와 같다.  
이러한 W값에서 어떤 음수만큼 빼주게 되어 더하게 된다.  
$$W(update)=w-a\frac{\partial f_c(x)}{\partial W}$$  
즉 W값이 점점 커지면서 새롭게 갱신된 W에 대해서 위와같은 공식을 반복적으로 적용한다.  
<span style ="color: red">**주의해야 할 점은 a(학습률 파라미터 = Learning Rate)를 적절한 값으로 설정해줘야 한다는 것이다.**</span><br>
학습률 파라미터가 너무 작은 값이면 최적의 w를 찾아가는데 너무 오래 걸릴 가능성이 크고, 너무 크면 최적의 지점을 건너뛰어 버리고 발산해 버릴 수 있다.  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/3.PNG" height="250" width="600" /></div>
계속하여 W를 갱신하여 Cost값이 최소가 되는(미분값이 0 인) 곳을 찾는다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/AI/4.PNG" height="250" width="600" /></div>
<hr>
참조: <a href="https://www.youtube.com/watch?v=GmtqOlPYB84&list=PL1H8jIvbSo1q6PIzsWQeCLinUj_oPkLjc&index=21">Chanwoo Timothy Lee Youtube</a> <br>
참조: <a href="https://honeytip91.tistory.com/106">honeytip91 블로그</a> <br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
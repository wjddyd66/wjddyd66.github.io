---
layout: post
title:  "Perceptron"
date:   2019-07-26 09:00:00 +0700
categories: [DL]
---

### Perceptron
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>

Perceptron은 다수의 신호(Input)을 입력받아서 하나의 신호(Output)을 출력한다. 이는 뉴런이 전기신호를 내보내 정보를 전달하는 것과 비슷하다.  
뉴런의 신호전달하는 역할을 Perceptron에서는 weight  
신호의 총합이 정해진 입계값(<span>$$\theta: 세타$$ </span>)를 넘었을때 1을 출력한다.  
<div><img src="https://image.slidesharecdn.com/lecture29-convolutionalneuralnetworks-visionspring2015-150504114140-conversion-gate02/95/lecture-29-convolutional-neural-networks-computer-vision-spring2015-9-638.jpg?cb=1430740006" height="250" width="800" /></div>

그림출처<a href="https://www.slideshare.net/jbhuang/lecture-29-convolutional-neural-networks-computer-vision-spring2015">Jia-Bin Huang PPT</a>  
<br>
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">

	<tr bgcolor="silver">	
		<th>실제 값</th>
		<th>용어</th>
	</tr>
	
	<tr>
		<td>x1, x2, ..., xd</td><td>Input(입력)</td>
	</tr>
	
	<tr>
		<td>w1, w2, ..., wd</td><td>Weight(가중치)</td>
	</tr>
	<tr>
		<td>y</td><td>Output(출력)</td>
	</tr>
	<tr>
		<td>b</td><td>bias(편향)</td>
	</tr>
	<tr>
		<td>$$ \sigma $$</td><td>Activatation Function(활성화 함수)</td>
	</tr>
</table>
<br>

Activate Function(Input과 가중치를 활용하여 Output을 만들어내기 위한 식)이 Sigmoid를 사용하므로 최종적인 Output의 식은 아래와 같다.  

<p> $$ y = {1 \over 1+e^{-\sum_{i=0}^d w_ix_i + b}}$$ </p>
최종적으로 우리는 위의 식에 대입한 값이 <span>$$\theta$$ </span>를 넘으면 1, 넘지 못하면 0으로서 판단할 수 있게 된다.  
실제 y와 x는 주어진 Data이므로 우리가 중점적으로 봐야할 것은 **weight 와 bias** 이다.  
적절한 **weight 와 bias** 를 조정하기 위한 과정이 반복적으로 이루어지게 되고 우리는 이러한 것을 'Model을 **Trainning** 한다' 라고 표현한다.  
이러한 Trainning 하여 알아내야 되는 변수의 의미를 살펴보면 아래와 같다.  
<span style ="color: red">**가중치(weight)는 입력신호가 결과 출력에 주는 영향도를 조절하는 매개변수이고, 편향(bias)은 뉴런(또는 노드; x를 의미)이 얼마나 쉽게 활성화(1로 출력; activation)되느냐를 조정하는(adjust) 매개변수이다.**</span>  

### Perceptron의 한계점과 극복
Perceptron의 한계점으로는 <span style ="color: red">**선형분류만 가능**</span>하다는 것이다.  
XOR과 같이 선형분류가 아닌 exclusive 논리연산은 분류할 수 없다는 문제가 생기게 된다.  
<div><img src="http://ecee.colorado.edu/~ecen4831/lectures/xor2.gif" height="250" width="800" /></div>

그림출처<a href="http://ecee.colorado.edu/~ecen4831/lectures/NNet3.html">ecee.colorado.edu</a>  
**AND Gate 진리표**  
<table class="table">

	<tr bgcolor="silver">	
		<th>X1</th>
		<th>X2</th>
		<th>Y</th>
	</tr>
	
	<tr>
		<td>0</td><td>0</td><td>0</td>
	</tr>
	
	<tr>
		<td>0</td><td>1</td><td>0</td>
	</tr>
	<tr>
		<td>1</td><td>0</td><td>0</td>
	</tr>
	<tr>
		<td>1</td><td>1</td><td>1</td>
	</tr>
</table>
<br>
**AND Gate 구현**  
```python
#AND
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp<0:
        return 0
    else:
        return 1
```
**AND Gate 결과 확인**  
```python
#AND 확인
print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))
```
```code
0
0
0
1
```

**OR Gate 진리표**  
<table class="table">

	<tr bgcolor="silver">	
		<th>X1</th>
		<th>X2</th>
		<th>Y</th>
	</tr>
	
	<tr>
		<td>0</td><td>0</td><td>0</td>
	</tr>
	
	<tr>
		<td>0</td><td>1</td><td>1</td>
	</tr>
	<tr>
		<td>1</td><td>0</td><td>1</td>
	</tr>
	<tr>
		<td>1</td><td>1</td><td>1</td>
	</tr>
</table>
<br>
**OR Gate 구현**  
```python
#OR
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp<0:
        return 0
    else:
        return 1
```
**OR Gate 결과 확인**  
```python
#OR 확인
print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))
```
```code
0
1
1
1
```
<br>
위와 같이 <span style ="color: red">**선형분류만 가능**</span>하다는 한계점을 극복하기 위하여 **다층 퍼셉트론**을 이용하여 한계를 극복하였다.  
다층 Perceptron이란 **여러개의 Perceptron**으로 인하여 **비선형 영역을 분리할 수 없다는 한계를 해결**한 것이다.  
<div><img src="https://upload.wikimedia.org/wikipedia/commons/b/b2/Perceptron_XOR.jpg" height="250" width="800" /></div>

그림출처<a href="https://upload.wikimedia.org/wikipedia/commons/b/b2/Perceptron_XOR.jpg">ecee.colorado.edu</a>  
<br>
**XOR Gate 진리표**  
<table class="table">

	<tr bgcolor="silver">	
		<th>X1</th>
		<th>X2</th>
		<th>Y</th>
	</tr>
	
	<tr>
		<td>0</td><td>0</td><td>0</td>
	</tr>
	
	<tr>
		<td>0</td><td>1</td><td>1</td>
	</tr>
	<tr>
		<td>1</td><td>0</td><td>1</td>
	</tr>
	<tr>
		<td>1</td><td>1</td><td>0</td>
	</tr>
</table>
<br>
**XOR Gate 구현**  
```python
#MLP
#XOR
def XOR(x1,x2):
    s1 = lambda x1,x2:1 if AND(x1,x2) == 0 else 0
    s2 = OR(x1,x2)
    y = AND(s1(x1,x2),s2)
    return y
```
**XOR Gate 결과 확인**  
```python
#XOR 확인
print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))
```
```code
0
1
1
0
```
<hr>
참조: <a href="https://github.com/wjddyd66/DeepLearning/blob/master/Perceptron.ipynb">원본코드</a><br>
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
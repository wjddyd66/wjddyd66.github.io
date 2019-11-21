---
layout: post
title:  "Matplotlib"
date:   2019-07-29 10:01:00 +0700
categories: [DataAnalysis]
---

### Matplotlib
**Matplotlib**이란 다양한 데이터를 많은 방법으로 도식화 할 수 있도록 하는 파이썬 라이브러리로써, 주로 matplotlib 의 pyplot을 사용한다.  
**Matplotlib**을 사용하게 되면 이전에 사용하였던 **numpy**나 **pandas**에서 사용되는 자료 구조를 쉽게 시각화 할 수 있다.  

#### Load Package
```python
import numpy as np
import matplotlib.pyplot as plt
```

#### Matplotlib 기본 속성
Matplotlib 에서는 다음과 같은 기본 속성을 가지고 있다.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
<thead><tr>
<th>속성</th>
<th>종류</th>
</tr>
</thead>
<tbody>
<tr>
<td>alpha</td>
<td>투명도</td>
</tr>

<tr>
<td>kind</td>
<td>
투명도: line, bar, barh, kde
</td>
</tr>

<tr>
<td>logY</td>
<td>Y축에 대해 Log scaling</td>
</tr>

<tr>
<td>use_index</td>
<td>객체의 색인을 눈금 이름으로 사용할 지 여부</td>
</tr>

<tr>
<td>rot</td>
<td>눈금 이름 돌리기(rotating)0~360</td>
</tr>

<tr>
<td>xticks, yticks</td>
<td>x, y축으로 사용할 값</td>
</tr>

<tr>
<td>xlim, ylim</td>
<td>x, y 축의 한계</td>
</tr>

<tr>
<td>grid</td>
<td>축의 그리드를 표현할지 여부</td>
</tr>

<tr>
<td>subplots</td>
<td>각 column에 독립된 subplot 그리기</td>
</tr>

<tr>
<td>sharex, sharey</td>
<td>subplots = True이면 같은 x, y축을 공유하고 눈금과 한계를 연결</td>
</tr>

<tr>
<td>figsize</td>
<td>생성될 그래프의 크기를 tuple로 지정</td>
</tr>

<tr>
<td>title</td>
<td>그래프의 제목 지정</td>
</tr>

<tr>
<td>legend</td>
<td>subplot의 범례 지정</td>
</tr>

<tr>
<td>sort_columns</td>
<td>column을 알파벳 순서로 그린다.</td>
</tr>

</tbody>
</table>
<br>

#### 점선 그리기
**Data 생성**  
```python
data = np.random.randn(50).cumsum()
data
```

```code
array([-0.52643756, -0.43216741, -0.64660902,  0.66810856,  1.26133996,
        0.51176633, -2.06579121,  0.13919543,  0.64501733,  0.17811056,
        1.73930986,  3.49944408,  2.49028208,  1.85480593,  2.45579842,
        2.67333743,  2.63689118,  2.89998175,  2.1737935 ,  1.80671599,
        2.69086579,  2.62257335,  1.11423586,  2.19940705,  2.61943852,
        1.72773764,  1.92567366,  2.67938229,  2.58160257,  3.29155193,
        3.90138574,  5.05262044,  4.9761306 ,  5.18704293,  4.32595818,
        4.83653822,  5.78111082,  6.09278176,  7.30988381,  9.0688758 ,
        6.57130842,  5.00812931,  5.02239784,  3.93325657,  2.5465875 ,
        0.86834185,  0.43049766, -0.59775005,  0.47730353, -0.0417067 ])
```


- plt.plot(): 그래프 그리기
- plt.show(): 그래프 보기

```python
plt.plot(data)
plt.show()
```

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/1.PNG" height="50%" width="50%" /></div><br>

#### 여러 그래프 그리기
**plt.subplot("행", "열", "순서")**  
```python
plt.subplot(1,2,1)
plt.subplot(1,2,2)
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/2.PNG" height="50%" width="50%" /></div><br>
**Data 준비하기**  

```python
hist_data = np.random.randn(100)
scat_data = np.arange(30)
```

**여러 그래프 그리기**  
```python
plt.subplot(2,2,1)
plt.plot(data)
plt.subplot(2,2,2)
plt.hist(hist_data, bins=20)
plt.subplot(2,2,3)
plt.scatter(scat_data,np.arange(30) +3 )
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/3.PNG" height="50%" width="50%" /></div><br>
#### 그래프 선 옵션
그래프를 그릴 때 표시되는 색이나 마커 패턴을 바꾸는 것을 확인  
- 색상: b(파란색), g(초록색), r(빨간색), c(청록색), y(노란색), k(검은색), w(흰색)
- 마커: o(원), v(역삼각형), ^(삼각형), s(네모), +(플러스), .(점)


```python
plt.plot(data,'g^')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/4.PNG" height="50%" width="50%" /></div><br>
```python
plt.plot(data,'+')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/5.PNG" height="50%" width="50%" /></div><br>

#### 그래프 사이즈 조절
plt.figure 안에 figsize를 이용하여 가로, 세로 길이 조절 가능(inch 단위)  
subplot과 같이 사용 시 맨 위에 있어야 전부다 적용 가능  
```python
plt.figure(figsize=(10,5))
plt.plot(data,'k+')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/6.PNG" height="50%" width="50%" /></div><br>
여러 그래프를 그리고 그에 대한 크기 조절  
```python
plt.figure(figsize=(10,5))
plt.subplot(2,2,1)
plt.plot(data)
plt.subplot(2,2,2)
plt.hist(hist_data, bins=20)
plt.subplot(2,2,3)
plt.scatter(scat_data,np.arange(30) +3 )
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/7.PNG" height="50%" width="50%" /></div><br>
#### 그래프 겹치기  + legend 달기
```python
data = np.random.randn(30).cumsum()

plt.plot(data, 'k--', label='Default')
plt.plot(data, 'k--',drawstyle='steps-post', label='steps-post')
plt.legend()
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/8.PNG" height="50%" width="50%" /></div><br>
#### 이름 달기
```python
plt.plot(np.random.randn(1000).cumsum())
plt.title('Random Graph')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/9.PNG" height="50%" width="50%" /></div><br>
#### 종합
```python
plt.title('Graph')
plt.plot(np.random.randn(1000).cumsum(),'k',label='one')
plt.plot(np.random.randn(1000).cumsum(),'b-',label='two')
plt.plot(np.random.randn(1000).cumsum(),'r^',label='three')

plt.legend()
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/10.PNG" height="50%" width="50%" /></div><br>
#### 그래프 저장하기
```python
plt.savefig('saved_graph.svg')
```
```code
<Figure size 432x288 with 0 Axes>
```

#### 이미지 파일 열기
opencv로도 열 수 있지만, shape 순서가 바뀔 때도 있다.
```python
path = './programmer.png'
image_pil = Image.open(path)
image = np.array(image_pil)
image.shape
```
```code
(512, 512, 4)
```

#### 이미지 정보 확인
image의 크기, 최대, 최소 값을 확인한다.
```python
image.shape
np.min(image), np.max(image)
```
```code
(512, 512, 4)
(0, 255)
```

#### 이미지 그래프로 시각화 하기
```python
plt.hist(image.ravel(),256,[0,256])
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/11.PNG" height="50%" width="50%" /></div><br>

#### 그림 나타내기
```python
plt.imshow(image)
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/12.PNG" height="50%" width="50%" /></div><br>

#### 이미지 흑백으로 열기
Image.convert("L")을 통하여 회색으로 변환시키는 것이다.
```python
image_pil = Image.open(path).convert("L")
image_bw = np.array(image_pil)
plt.imshow(image_bw, 'gray')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/13.PNG" height="50%" width="50%" /></div><br>

#### 이미지 다른 색상으로 열기
**RdBu**
```python
plt.imshow(image_bw,'RdBu')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/14.PNG" height="50%" width="50%" /></div><br>
**jet**
```python
plt.imshow(image_bw,'jet')
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/15.PNG" height="50%" width="50%" /></div><br>
**Colorbar 추가하기**
```python
plt.imshow(image_bw,'jet')
plt.colorbar()
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/16.PNG" height="50%" width="50%" /></div><br>

#### 이미지 사이즈 조절
```python
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/17.PNG" height="50%" width="50%" /></div><br>

#### 이미지에 제목 추가
```python
plt.title('Programmers')
plt.imshow(image)
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/18.PNG" height="50%" width="50%" /></div><br>
#### 두번째 이미지 열기 및 이미지 크기 변환
```python
path2 = './bomair_logo.png'

image2_pil = Image.open(path2)
image2 = np.array(image2_pil)

plt.imshow(image2)
plt.show()

import cv2
image3 = cv2.resize(image2, (512,512))
image3.shape, image.shape
```
```code
((512, 512, 4), (512, 512, 4))
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/19.PNG" height="50%" width="50%" /></div><br>
#### 이미지 합치기
```python
plt.imshow(image)
plt.imshow(image3, alpha=0.5)
plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/20.PNG" height="50%" width="50%" /></div><br>
#### 이미지에 Subplot
```python
plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(image)
plt.subplot(2,2,2)
plt.imshow(image_bw, 'gray')
plt.subplot(2,2,3)
plt.imshow(image3)

plt.show()
```
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Python/21.PNG" height="50%" width="50%" /></div><br>
<hr>
참조: <a href="https://github.com/wjddyd66/DataAnalysis/blob/master/Scipy.ipynb">원본코드</a> <br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
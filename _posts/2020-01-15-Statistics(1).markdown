---
layout: post
title:  "Statistics(1)-Basic"
date:   2020-01-20 09:10:20 +0700
categories: [statistics]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
이번 POST는 <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업 내용</a>을 정리한 것 입니다.  
문제나 자세한 내용은 <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a>를 참조하였습니다.  

### 1. 확률(Statistics)
통계학이란 관심의 대상이 되는 집단으로부터 자료를 수집, 정리, 요약하여 제한된 자료나 정보를 토대로 불확실한 사실에 대하여 과학적인 판단을 내릴 수 있도록 그 방법과 절차를 제시하여 보여주는 학문이다.  
<br>

#### (1) 표본공간과 사상
1. 확률실험(random experimnet): 실험의 결과를 예측할 수 없는 실험 ex) 동전 던지기
2. 표본공간(sample space: S): 통계적인 실험에서 발생 가능한 서로 다른 모든 결과의 집합 ex) 동전 던지기 -> S = {H(앞면),T(뒷면)}
3. 사상(event): 표본공간의 부분집합을 의미한다. ex) A = {H}(동전을 던졌을 경우 앞면이 나올 사상), B = {T}(동전을 던졌을 경우 뒷면이 나올 사상)
4. 배반사상(exclusive event), 상호 배반(mutally exclusive): 두 개의 사상 A 와 B 를 나타내는 부분집합들이 서로 동일한 근원사상을 포함하고 있지 않는 경우를 의미한다. <span>$$A \cap B = \emptyset$$</span>
5. 합집합(union): <span>$$A \cup B$$</span>
6. 교집합(intersection): <span>$$A \cap B$$</span>
7. 여집합(complement): <span>$$\bar{A}, A^{'}, A^{c}$$</span>

#### (2) 확률의 정의
<span>$$P(A) = \frac{n(A)}{n(S)}$$</span>: n(A)=사상A에 속하는 원소의 수, n(S)=표본공간에 속하는 원소의 수  
ex) 주사위 던지기 -> S = {1,2,3,4,5,6}, 홀수가 나올 경우의 수 -> A = {1,3,6}  
<span>$$P(A) = \frac{n(A)}{n(S)} = \frac{3}{6} = \frac{1}{2}$$</span>

#### (3) 확률의 공리
1. <span>$$0 \le P(A) \le 1$$</span>
2. <span>$$P(S)=1$$</span>
3. 만약 <span>$$A_1, A_2, ....$$</span>이 상호 배반적이면, <span>$$P(A_1 \cup A_2 \cup ...) = P(A_1) + P(A_2) + ...$$</span>

#### (4) 조건부 확률
1. 사상 B가 일어난다는 조건하에서 사상 A가 일어날 확률 <span>$$P(A|B)=\frac{P(A \cap B)}{P(B)} \text{ 단, }P(B)\ge 0$$</span>
2. <span>$$P(A) \ge 0, P(B) \ge 0 \text{ 일 때 } P(A \cap B) = P(B|A)P(A) = P(A|B)P(B)$$</span>
3. 만약 사상 A, B가 서로 독립이면 <span>$$P(A \cap B)=P(A)P(B)$$</span>, <span>$$P(A|B)=\frac{P(A \cap B)}{P(B)} = \frac{P(A)P(B)}{P(B)} = P(A)$$</span>

#### (5) 독립
<span>$$P(A \cap B)=P(A)P(B)$$</span>이면 독립이다.  
ex) <span>$$S = {1,2,3,4,5,6}, A={1,2,4}, B={3,4}$$</span>
<p>$$P(A)=\frac{1}{2}, P(B)=\frac{1}{3}, P(A \cap B) = \frac{1}{6}$$</p>
<p>$$P(A)P(B) = P(A \cap B) \therefore \text{A,B는 서로 독립이다.}$$</p>
<p>$$\text{단, 상호배반은 아니다.} \because P(A \cap B) = \emptyset$$</p>
A, B, C가 상호 독립이라면 다음과 같은 조건을 검사하여야 한다.  
1) <span>$$P(A,B) = P(A)P(B)$$</span>  
2) <span>$$P(A,C) = P(A)P(C)$$</span>  
3) <span>$$P(B,C) = P(B)P(C)$$</span>  
4) <span>$$P(A,B,C) = P(A)P(B)P(C)$$</span>  
<br>

ex) S={(1,0,0),(0,1,0),(0,0,1),(1,1,1)}  
<p>$$P(A) = \frac{1}{2}\text{ (x가 1일 확률), } P(B) = \frac{1}{2}\text{ (y가 1일 확률), } P(C) = \frac{1}{2}\text{ (z가 1일 확률)}$$</p>
<p>$$P(A,B) = \frac{1}{4} = P(A)P(B) = \frac{1}{2}*\frac{1}{2}$$</p>  
<p>$$P(A,B,C) = \frac{1}{4} = P(A)P(B)P(C) \neq \frac{1}{2}*\frac{1}{2}*\frac{1}{2}$$</p>
<p>$$\therefore \text{A, B, C 는 상호 독립이 아니다.}$$</p>

#### (6) 곱셈법칙
원소의 개수 <span>$$n_1, n_2, ..., n_k$$</span>인 집합 <span>$$A_1, A_2, ..., A_k$$</span>에서 각각 하나의 원소를 택하여 나열한 순서열의 개수는 <span>$$n_1 * n_2 * n_3 * ....$$</span>

#### (7) 순열과 조합
순열: n개의 원소 중에서 r개의 원소를 **순서를 고려해서 선택**: <span>$$_{n}\mathrm{P}_{r} = \frac{n!}{(n-r)!}$$</span>  
ex) <span>$$_{4}\mathrm{P}_{2} = \frac{4!}{(4-2)!} = 12$$</span>  
조합: n개의 원소 중에서 r개의 원소를 **순서를 고려하지 않고 선택**: <span>$$_{n}\mathrm{C}_{r} = \frac{n!}{(n-r)!r!}$$</span>  
ex) <span>$$_{4}\mathrm{C}_{2} = \frac{4!}{(4-2)!2!} = 6$$</span>  

**주의 사항**  
1) <span>$$_{n}\mathrm{C}_{n} = \frac{n!}{(n-n)!n!} = \frac{1}{0!} = 1$$</span>  
2) <span>$$_{n}\mathrm{C}_{0} = \frac{n!}{(n-0)!0!} = \frac{1}{0!} = 1$$</span>  
3) <span>$$_{n}\mathrm{C}_{k} = 0 (n > k)$$</span>

#### (8) 베이즈 정리
사상 A가 일어났다고 가정했을 때 사상 B가 일어날 확률을 P(B|A)라 표시하고, 이를 사상 A를 조건으로 하는 B의 조건부 확률이라고 한다면, 사상 <span>$$B_1, B_2, ..., B_k$$</span>를 표본공간 S의 분할이라고 할 때 임의의 사상 A가 나타난 후에 특정 사상 <span>$$B_j$$</span>에 속할 확률을 다음과 같이 정의하며, 이것을 **베이즈 정리**라고 한다.  
<p>$$P(A_j|B) = \frac{P(A_j \cap B)}{\sum_{i=1}^{k}P(A_i \cap B)} = \frac{P(A_j)P(B|A_j)}{\sum_{i=1}^{k}P(A_i)P(B|A_i)}$$</p>
<p>$$\because P(B) = \sum_{i=1}^{k}P(A_i \cap B)$$</p>
<p>$$P(A_j \cap B) = P(A_j)P(B|A_j)$$</p>
<img src="http://i.imgur.com/jC7FfHv.png"/><br>
사진 출처: <a href="https://ratsgo.github.io/statistics/2017/07/01/bayes/">ratsgo's blog</a><br>

ex) 제품을 A사가 30%, B사가 30%, C사가 40%를 생산할 때 각각 회사의 불량품이 나올 확률은 1%, 1%, 0.5%라고 한다. 불량품을 선택하였을 때 A사의 불량품일 확률을 구하여라.  
<p>$$P(A) = 0.3, P(B) = 0.3, P(C) = 0.4$$</p>
<p>$$P(D|A) = 0.01, P(D|B) = 0.01, P(D|C) = 0.005$$</p>
<p>$$P(A|D) = \frac{P(A \cap D)}{P(D)} = \frac{P(A \cap D)}{P(A \cap D) + P(B \cap D) + P(C \cap D)}$$</p>
<p>$$P(D|A) = \frac{P(A \cap D)}{P(A)} = 0.01$$</p>
<p>$$\therefore P(A \cap D) = 0.3*0.01 = 0.003$$</p>
<p>$$\therefore P(A|D) = \frac{0.003}{0.003+0.003+0.002} = \frac{3}{8}$$</p>
<br><br>

### 2. 확률 변수
확률변수 X는 표본공간에 속하는 원소를 실수(real number)로 변환시키는 함수이다.  

#### (1) 이산확률변수(discrete random variable)
유한개의 값을 취하거나, 일정 구간내의 실수값이 아무리 많더라도 하나하나 셀 수 있는 확률변수를 뜻하는 것으로, 계수치가 이산형 확률변수가 된다.  

**확률함수 f(x)의 성질**  
1. <span>$$f(x) \ge 0$$</span>
2. <span>$$\sum f(x) = 1$$</span>
3. <span>$$f(a \le x \le b) = \sum_{x=a}^{b} f(x)$$</span>

ex) 동일한 동전 3회 던지는 경우 x = H의 수  
<p>$$R_x = {0,1,2,3}$$</p>
<p>$$P(X=0) = {}_{3}\mathrm{C}_{0} * \frac{1}{8} = \frac{1}{8}$$</p>
<p>$$P(X=1) = {}_{3}\mathrm{C}_{1} * \frac{1}{8} = \frac{3}{8}$$</p>
<p>$$P(X=2) = {}_{3}\mathrm{C}_{2} * \frac{1}{8} = \frac{3}{8}$$</p>
<p>$$P(X=3) = {}_{3}\mathrm{C}_{3} * \frac{1}{8} = \frac{1}{8}$$</p>
누적확률 함수 <span>$$F(x) = P(X \le x)$$</span>의 성질  
1. <span>$$F(-\infty) = 0, F(\infty) = 1$$</span>
2. 비감소(non-decreasing) 함수
3. continuous from the right 함수

위의 누적확률 함수를 그래프로 표현하면 다음과 같다.  
<img src="https://mblogthumb-phinf.pstatic.net/20161013_212/mykepzzang_14763538246355BrCx_JPEG/%C7%C1%B7%B9%C1%A8%C5%D7%C0%CC%BC%C71.jpg?type=w2" largesrc="javascript:location.href='https://mblogthumb-phinf.pstatic.net/20161013_212/mykepzzang_14763538246355BrCx_JPEG/%C7%C1%B7%B9%C1%A8%C5%D7%C0%CC%BC%C71.jpg?type=w2'"/><br>
사진 출처: <a href="https://m.blog.naver.com/mykepzzang/220835517006">mykepzzang 블로그</a><br>
오른쪽에서의 극한값은 Continous하지만 왼쪽에서의 극한값은 Continuous하지 않는다.

#### (2) 연속확률변수(continuous random variable)
제품의 크기나 중량처럼 일정 구간내의 실수값이 무한갯수로 정의되는 확률변수를 뜻하는 것으로, 계량치가 연속형 롹률변수의 값이 된다.  

**밀도함수 f(x)의 성질**  
1. <span>$$f(x) \ge 0$$</span>
2. <span>$$\int_{-\infty}^{\infty} f(x)\, dx = 1$$</span>
3. <span>$$f(a \le x \le b) = \int_{a}^{b} f(x)\, dx$$</span>

누적확률 함수 <span>$$F(x) = P(X \le x)$$</span>의 성질  
1. <span>$$F(-\infty) = 0, F(\infty) = 1$$</span>
2. 증가(increasing) 함수
3. 연속(continuous) 함수

**주의 사항**  
P(x) = 이상형 확률 변수, Q(x) = 연속형 확률 변수라 하면  
<p>$$P(x \le a) \neq P(x < a)$$</p>
<p>$$Q(x \le a) = Q(x < a)$$</p>
#### (3) 이산확률변수의 결합확률분포(Joint Probability Distribution of Discrete Random Variable)
두 개의 이산확률변수 X와 Y가 각각 <span>$$x_1, x_2, ...$$</span>와 <span>$$y_1, y_2, ...$$</span>의 값을 가질때 <span>$$P(X=x,Y=y) = f(x,y)$$</span>를 만족하는 <span>$$f(x,y)$$</span>를 이산확률변수 X와 Y의 **결합확률분포**또는 **결합확률질량함수**라고 한다.  

**결합확률분포의 성질**  
1. 모든 (x,y)에 대하여 <span>$$f(x,y) \ge 0$$</span>
2. <span>$$\sum_{x} \sum_{y} f(x,y) = 1$$</span>
3. x,y 평면 상의 어떤 A에 대해 <span>$$P[(x,y) \in A] = \sum \sum_{A} f(x,y)$$</span>

ex) 3개의 검은 구슬, 2개의 붉은 구슬, 3개의 흰 구슬이 들어있는 상자에서 임의로 2개의 구슬을 꺼낼 때, 검은 구슬의 개수를 X, 붉은 구슬의 개수를 Y라 하면 X와 Y의 결합확률 분포를 구하여라. 또 <span>$$A = [(x,y)|(x+y \le 1)]$$</span>일 때 <span>$$P[(x,y) \in A]$$</span>를 구하여라.  

<p>$$R_x = {0,1,2,3}, R_Y = {0,1,2}, R_x + R_y \le 2$$</p>
<p>$$\therefore R_x + R_y = [(0,0),(1,0),(0,1),(2,0),(0,2),(1,1)]$$</p>
<p>$$f(0,0) = \frac{_{3}\mathrm{C}_{2}}{_{8}\mathrm{C}_{2}}$$</p>
<p>$$= \frac{3}{28} \text{ 전체 구슬 8개중 2개를 선택하는 경우 흰구슬 3개에서 2개를 모두 선택}$$</p>
<p>$$f(0,1) = \frac{_{3}\mathrm{C}_{2} * _{2}\mathrm{C}_{1}}{_{8}\mathrm{C}_{2}}$$</p>
<p>$$= \frac{6}{28} \text{ 전체 구슬 8개중 2개를 선택하는 경우 흰 구슬 3개에서 1개를 선택, 붉은 구슬2개중 1개를 선택}$$</p>
위와 같이 모든 경우의 수를 계산하면 아래와 같은 결합확률 분포표를 얻을 수 있다.  
<table>
    <tr>
        <td>Y\X</td><td>0</td><td>1</td><td>2</td>   
    </tr>
    <tr>
        <td>0</td><td><span>$$\frac{3}{28}$$</span></td><td><span>$$\frac{9}{28}$$</span></td><td><span>$$\frac{3}{28}$$</span></td>   
    </tr>
    <tr>
        <td>1</td><td><span>$$\frac{6}{28}$$</span></td><td><span>$$\frac{6}{28}$$</span></td><td>0</td>   
    </tr>
    <tr>
        <td>2</td><td><span>$$\frac{1}{28}$$</span></td><td>0</td><td>0</td>   
    </tr>
</table>
위의 결합분포표를 활용하여 <span>$$P[(x,y) \in A]$$</span>를 계산하게 되면 다음과 같다.  
<p>$$P[(x,y) \in A] = f(0,0) + f(0,1) + f(1,0) = \frac{9}{14}$$</p>
#### (4) 연속확률변수의 결합밀도 함수(Joing Density Function of Continuous Random Variable)
결합밀도 함수는 연속확률변수가 두 개 이상인 확률밀도함수 이다.  
**결합밀도함수의 성질**  
1. 모든 (x,y)에 대하여 <span>$$f(x,y) \ge 0$$</span>
2. <span>$$\int_{-\infty}^{\infty}\, \int_{-\infty}^{\infty}\, f(x,y) dydx = 1$$</span>
3. x,y 평면 상의 어떤 A에 대해 <span>$$P[(x,y) \in A] = \int\, \int_{A}\, f(x,y) dxdy$$</span>를 만족하는 f(x,y)를 결합확률밀도합수 라고 한다.  

ex) 결합밀도 함수가 <span>$$f(x,y) = e^{-x-y} (x \ge 0, y \ge 0)$$</span>일 때, <span>$$P(X+Y \le 1)$$</span>을 구하여라.  

먼저 적분을 하기 위하여 X,Y의 범위를 알아내기 위하여 확률영역을 그리게 되면 다음과 같다.  
<img src="https://mblogthumb-phinf.pstatic.net/20161015_184/mykepzzang_14764750559424wLgb_JPEG/%C7%C1%B7%B9%C1%A8%C5%D7%C0%CC%BC%C71.jpg?type=w420"/><br>
사진 출처: <a href="https://m.blog.naver.com/mykepzzang/220836634095">mykepzzang 블로그</a><br>
<p>$$0 \ge X \le 1, Y = 1-X$$</p>
위에서 구한 범위를 결합밀도 합수에 적용하게 되면  
<p>$$\int_{0}^{1}\, \int_{0}^{1-x}\, e^{-x-y} dydx$$</p>
<p>$$= \int_{0}^{1}\, [-e^{-x-y}]_0^{1-x} dx = \int_{0}^{1}\, e^{-x}-e^{-1} dx$$</p>
<p>$$= [-e^{-x} -e^{-1}x]_0^1 = 1-\frac{2}{e}$$</p>
#### (5) 주변확률분포(Marginal Probability Distribution)
두 개의 변수로 이루어진 결합확률분포를 통해 하나의 변수로만 이루워진 확률함수를 구하는 것  
즉, X,Y 두 개의 확률변수로 이루어진 함수를 X또는 Y의 하나의 확률변수로 표현하기 위해서 주변확률분포를 이용한다.  
주변확률분포의 정의는 다음과 같다.  
결합확률 분포 f(x,y)가 확률변수 X 또는 Y만의 분포이면  
(1) 확률변수가 이산확률변수일 경우  
<p>$$f_X(x) = \sum_y f(x,y)$$</p>
<p>$$f_Y(y) = \sum_x f(x,y)$$</p>
(2) 확률변수가 연속확률변수일 경우  
<p>$$f_X(x) = \int_{-\infty}^{\infty} f(x,y)\, dy$$</p>
<p>$$f_Y(y) = \int_{-\infty}^{\infty} f(x,y)\, dx$$</p>
위에서 이산확률변수의 결합확률분포로부터 얻은 결합분포표를 주변확률분포표로 변형하면 다음과 같다.  

<table>
    <tr>
        <td>Y\X</td><td>0</td><td>1</td><td>2</td><td><span>$$f_Y(y)$$</span></td>   
    </tr>
    <tr>
        <td>0</td><td><span>$$\frac{3}{28}$$</span></td><td><span>$$\frac{9}{28}$$</span></td><td><span>$$\frac{3}{28}$$</span></td><td><span>$$\frac{15}{28}$$</span></td>   
    </tr>
    <tr>
        <td>1</td><td><span>$$\frac{6}{28}$$</span></td><td><span>$$\frac{6}{28}$$</span></td><td>0</td><td><span>$$\frac{12}{28}$$</span></td>   
    </tr>
    <tr>
        <td>2</td><td><span>$$\frac{1}{28}$$</span></td><td>0</td><td>0</td><td><span>$$\frac{1}{28}$$</span></td>   
    </tr>
    <tr>
        <td><span>$$f_X(x)$$</span></td><td><span>$$\frac{10}{28}$$</span></td><td><span>$$\frac{15}{28}$$</span></td><td><span>$$\frac{3}{28}$$</span></td><td>1</td>   
    </tr>
</table>

**만약 <span>$$f(x,y) = f_X(x)*f_Y(y)$$</span>이면 X,Y는 서로 독립이다.**

<br><br>

### 3. 확률변수의 평균과 분산
평균 or 기댓값(Expected Value, E(x)): 분포의 중심  
분산(Variation, V(x)): 평균으로부터 흩어짐의 척도  

#### (1) 이산형
1. <span>$$E(X) = \sum_{X} xP(X) = \mu$$</span>
2. <span>$$V(X) = \sum_{X}(X-\mu)^2 P(X) = \sum_{X} X^2 P(X) - (\sum_{X} X P(X))^2 = E(X^2) - (E(X))^2 = \sigma^2$$</span>

#### (2) 연속형
1. <span>$$E(X) = \int_{-\infty}^{\infty} Xf(X)\, dX$$</span>
2. <span>$$V(X) = \int_{-\infty}^{\infty} (X-\mu)^2f(X)\, dX = \int_{-\infty}^{\infty} X^2f(X)\, dX - \mu^2 = \sigma^2$$</span>

ex) X와 확률함수 f(x)가 아래와 같을때 평균과 분산을 구하여라.  
<table>
    <tr>
        <td>X</td><td>0</td><td>1</td><td>2</td><td>3</td>
    </tr>
    <tr>
        <td>f(x)</td><td><span>$$\frac{1}{8}$$</span></td><td><span>$$\frac{3}{8}$$</span></td><td><span>$$\frac{3}{8}$$</span></td><td><span>$$\frac{1}{8}$$</span></td>
    </tr>
</table>

<p>$$E(X) = \sum_{X}XP(X) = 0*\frac{1}{8} + 1*\frac{3}{8} + 2*\frac{3}{8} + 3*\frac{1}{8} = \frac{3}{2}$$</p>
<p>$$V(X) = E(X^2) - (E(X))^2 = E(X^2) - \frac{9}{4}$$</p>
<p>$$E(X^2) = \sum_{X^2}XP(X) = 0*\frac{1}{8} + 1*\frac{3}{8} + 4*\frac{3}{8} + 9*\frac{1}{8} = 3$$</p>
<p>$$\therefore V(X) = 3 - \frac{9}{4} = \frac{3}{4}$$</p>

#### (3) 평균과 분산의 특징
**평균 특징**  
<p>$$E(a) = a, E(aX) = aE(X), E(aX \pm b)$$</p>
위의 평균 특징에서 <span>$$E(aX \pm b)$$</span>에 대해서 알아보자.  
<p>$$E(aX \pm b) = \sum_{X}(ax \pm b)f(x) = a\sum_{X}xf(x) \pm b\sum_{X}f(x) = aE(X) \pm b$$</p>
**분산 특징**  
<p>$$V(a) = 0, V(aX) = a^2 V(X), V(aX \pm b) = a^2 V(X)$$</p>
위의 평균 특징에서 <span>$$V(aX \pm b)$$</span>에 대해서 알아보자.  
<p>$$V(aX \pm b) = \sum_{X}(ax \pm b)^2 f(x) -(aE(X)+b)^2$$</p>
<p>$$= \sum_{X}(a^2 \pm 2abx + b^2) f(x) -(a^2(E(X))^2+2abE(X) + b^2)$$</p>
<p>$$= a^2E(X^2) \pm 2abE(X) + b^2 - a^2(E(X))^2 \mp 2abE(X) - b^2$$</p>
<p>$$= a^2(E(X^2)-(E(X))^2) = a^2V(X)$$</p>
#### (4) 공분산(Covariance)
**공분산(Covariance)은 두개의 확률변수의 관계를 보여주는 값**이다.  
즉, 확률변수 X와 Y가 같이 변하는 정도를 나타낸 값으로서 <span>$$Cov(X,Y) = E[(X-\mu_x)(Y-\mu_y)] = E(XY) - \mu_X E(Y) - \mu_Y E(X) + \mu_X \mu_Y = E(XY) - \mu_X \mu_Y = E(XY) - E(X)E(Y)$$</span>로서 표현한다.  

만약 같은 변순끼리의 공분산을 구해보면  
<p>$$Cov(X,X) = E(XX) - E(X)E(X) = E(X^2) - (E(X))^2$$</p>
위의 식으로서 결국 분산이라는 것을 알 수 있다.  
만약 X,Y가 독립일 경우 공분산을 구해보면  
<p>$$Cov(X,Y) = E(XY) - E(X)E(Y) = E(X)E(Y) - E(X)E(Y) = 0$$</p>
두 변수가 독립이면 공분산은 0이 된다.  

**공분산 특징**  
a,b,c,d가 임의의 실수라면  
1. Cov(aX,bY) = abCov(X,Y)
2. Cov(X+a,Y+b) = Cov(X,Y)
3. Cov(X,aX+b) = aVar(X)
4. Cov(aX+b, cX+d) = acVar(X)

위의 4가지 특징 중에서 1,4만 알아보자.  
1번  
<p>$$Cov(aX,bY) = E(aXbY)-E(aX)E(bY) = abE(XY) - abE(X)E(Y) = ab(E(XY)-E(X)E(Y)) = abCov(X,Y)$$</p>
4번  
<p>$$Cov(aX+b,cX+d) = E((aX+b)(cX+d))-E(aX+b)E(cX+d)$$</p>
<p>$$= E(acX^2+adX+bcX+bd)-(aE(X)+b)(cE(X)+d)$$</p>
<p>$$= acE(X^2)+adE(X)+bcE(X)+bd -ac(E(X))^2-adE(X)-bcE(x)-bd$$</p>
<p>$$= ac(E(X^2)-(E(X))^2)$$</p>
<p>$$= acVar(X)$$</p>
ex) 두 확률 변수 X와 Y의 결합확률밀도함수가 다음과 같이 주어졌다.  
<p>$$
f(x,y)=
\begin{cases}
8xy, & 0 \le y \le x, 0 \le x \le 1 \\
0, & \mbox{elsewhere}
\end{cases}
$$</p>
공분산을 구하시오  

<p>$$Cov(X,Y) = E(XY) - E(X)E(Y)$$</p>
<p>$$E(X) = \int xf_x(x)\, dx, E(Y) = \int yf_y(y)\, dy, E(XY) = \int \int xyf(x,y)\, dydx$$</p>
<p>$$f_x(X) = \int_{-\infty}^{\infty} f(x,y)\, dy = \int_{x}^{0} 8xy\, dy = 4x^3 (0 \le x \le 1)$$</p>
<p>$$f_y(Y) = \int_{-\infty}^{\infty} f(x,y)\, dx = \int_{1}^{y} 8xy\, dx = 4y(1-y^2) (0 \le y \le 1)$$</p>
<p>$$\therefore E(X) = \int_{0}^{1} 4x^4\, dx = \frac{4}{5}$$</p>
<p>$$\therefore E(Y) = \int_{0}^{1} 4y(1-y^2)\, dy = \frac{8}{15}$$</p>
<p>$$\therefore E(XY) = \int_{0}^{1} \int_{0}^{x} 8xy\, dydx = \frac{4}{9}$$</p>
<p>$$\therefore Cov(X,Y) = \frac{4}{9} - \frac{4}{5} - \frac{8}{15} = \frac{4}{255}$$</p>
#### (5) 상관계수(Correlation Coefficient)
**공분산의 경우에는 측정단위에 큰 영향을 받기 때문에 측정단위에 영향을 받지 않는 지표가 필요하고 이러한 값을 상관계수라고 표현한다.**  
두 확률변수 X,Y의 상관계수를 <span>$$\rho(X,Y) = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}, -1 \le \rho(X,Y) \le 1, \sigma_X, \sigma_Y \text{ 는 각각 X,Y의 표준편차}$$</span>
상관계수의 값에 따라 각각 다음과 같은 의미가 있다.  
- <span>$$\rho(X,Y) = 1$$</span>이면 X와 Y는 완전 비례관계
- <span>$$\rho(X,Y) = -1$$</span>이면 X와 Y는 완전 반비례관계
- <span>$$\rho(X,Y) = 0$$</span>이면 X와 Y는 서로 관련이 없음(독립)

위의 상황을 Visualization하면 아래 그림과 같다.  
<img src="https://mblogthumb-phinf.pstatic.net/MjAxNjEwMjlfMjI1/MDAxNDc3NjcyNDkwOTM5.KyM_SA1naGBolsQT_IVmgeNQ5NOgJ7IIEkIaZ78Cb-cg.W52kkCLHwXvILVTGjfmVbLsTcYPtyswwndV_eIHtp-Ig.JPEG.mykepzzang/IMG_3172.jpg?type=w2"/><br>
사진 출처: <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a><br>

<hr>
참조: <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업</a><br>
참조: <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


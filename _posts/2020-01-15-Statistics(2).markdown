---
layout: post
title:  "Statistics(2)-Discrete Probability Distributions"
date:   2020-01-20 09:20:20 +0700
categories: [statistics]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
이번 POST는 <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업 내용</a>을 정리한 것 입니다.  
문제나 자세한 내용은 <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a>를 참조하였습니다.  

### 4. 이산형 확률분포

#### (1) 베르누이분포(Bernouli distribution)
어느 실험 또는 관찰을 독립적으로 반복해서 시행하는 경우에 매 시행마다 오직 두 개의 결과만이 일어나며, 각 시행이 서로 독립적인 것을 베르누이 시행이라고 한다.  
**베르누이 확률변수의 확률질량함수 f(x)**  
<p>$$
f(x)=
\begin{cases}
p, & x=1 \\
q=(1-p), & x=0
\end{cases}
$$</p>
<p>$$f(x)=p^x (1-p)^{1-x}$$</p>
평균: <span>$$E(x) = \sum_{x=0}^{1} xf(x) = 0f(0)+f(1) = p$$</span>  
분산: <span>$$V(x) = \sum_{x=0}^{1} x^2f(x) - E(x)^2 = 0f(0)+f(1)-p^2 = p(1-p) =pq$$</span>

ex) 주사위를 한 번 던져 홀수가 나오면 실패, 짝수가 나오면 성공일 때 확률변수 X의 개댓값과 분산을 구하시오.  
위의 문제를 확률질량함수로서 표현하면 다음과 같다.  
<p>$$
f(x)=
\begin{cases}
\frac{1}{2}, & x=1 \\
\frac{1}{2}, & x=0
\end{cases}
$$</p>

<p>$$E(x) = p = \frac{1}{2}$$</p>
<p>$$V(x) = pq = \frac{1}{2} * \frac{1}{2} = \frac{1}{4}$$</p>
#### (2) 이항분포(Binomial distribution)
성공확률이 p인 경우 n번의 베르누이 실험에서 나타나는 성공횟수에 대한 확률변수 X가 따르는 분포를 이항분포라 한다. 이러한 이항분포를 따르는 확률변수 X의 확률함수는 아래와 같이 정의 된다. (단, <span>$$p \neq 0$$</span>)
<p>$$f(x) = {}_{n}\mathrm{C}_{x}p^x(1-p)^{n-x} = {}_{n}\mathrm{C}_{x}p^xq^{n-x}$$</p>
여기서 x=0,1,2,...,n 이다. (단, <span>$$_{n}\mathrm{C}_{x} = \frac{n!}{(n-x)!x!}$$</span>)  

평균: <span>$$E(x) = np$$</span>  
<p>$$E(x) = \sum_{r=0}^n r _{n}\mathrm{C}_{r} p^r(1-p)^{n-r} = \sum_{r=1}^n r_{n}\mathrm{C}_{r} p^r (1-p)^{n-r}$$</p>
<p>$$\because r=0 \rightarrow 0*{}_{n}\mathrm{C}_{0}p^0(1-p)^n = 0$$</p>
<p>$$= \sum_{r=1}^n r(\frac{n(n-1)!}{r!(n-r)!})pp^{r-1}(1-p)^{n-r}$$</p>
<p>$$=np\sum_{r=1}^n (\frac{(n-1)!}{(r-1)!(n-r)!})p^{r-1}(1-p)^{n-r}$$</p>
<p>$$=np\sum_{r=1}^n {}_{n-1}\mathrm{C}_{r-1}p^{r-1}(1-p)^{n-1}$$</p>
<p>$$=np\sum_{r=0}^{n-1}B(n-1,p) \text{  (단, B(n,p)는 확률인 p인 베르누이 실험을 n번한 이항분포)}$$</p>
<p>$$=np$$</p>
분산: <span>$$V(x) = npq$$</span>  
<p>$$V(x) = \sum_{r=0}^n r^2 f(r) - (np)^2 = \sum_{r=0}^n r^2{}_{n}\mathrm{C}_{r}p^r q^{n-r}  - (np)^2$$</p>
<p>$$= \sum_{r=1}^n r \frac{n(n-1)!}{(n-r)!(r-1)!}pp^{r-1}q^{n-r} - (np)^2$$</p>
<p>$$= np\sum_{r=1}^n r \frac{(n-1)!}{(n-r)!(r-1)!}p^{r-1}q^{n-r} - (np)^2$$</p>
<p>$$= np\sum_{r=1}^n r {}_{n-1}\mathrm{C}_{r-1}p^{r-1}q^{n-r} - (np)^2$$</p>
<br>
<p>$$\sum_{r=1}^n r {}_{n-1}\mathrm{C}_{r-1}p^{r-1}q^{n-r} = \sum_{s=0}^{n-1}(s+1) {}_{n-1}\mathrm{C}_{s}p^{s}q^{n-s-1} \because s=r-1$$</p>
<p>$$= \sum_{s=0}^{n-1}s {}_{n-1}\mathrm{C}_{s}p^{s}q^{n-s-1} + \sum_{s=0}^{n-1} {}_{n-1}\mathrm{C}_{s}p^{s}q^{n-s-1}$$</p>
위의 전개되어서 나온 식을 살펴보게 되면 각각은 다음과 같은 의미를 가지고 있다.  
<p>$$\sum_{s=0}^{n-1}s {}_{n-1}\mathrm{C}_{s}p^{s}q^{n-s-1}=B(n-1,p)\text{의 기대값}$$</p>
<p>$$\sum_{s=0}^{n-1} {}_{n-1}\mathrm{C}_{s}p^{s}q^{n-s-1}=B(n-1,p)\text{의 확률의 총합}$$</p>
<p>$$\therefore V(x) = np((n-1)p + 1) - (np)^2 = np(1-p) = npq$$</p>
ex) KBC리그에 뛰는 A선수는 타율이 2할7푼5리 이다. 이 선수가 어떤 경기에서 5버느이 타석에 들어섰을 때, 2개의 안타를 칠 확률은? (단, 각 타석의 결과는 서로 무관하다.)  
<p>$${}_{5}\mathrm{C}_{2}(\frac{275}{1000})^2(\frac{725}{1000})^3 = 0.2882$$</p>
#### (3) 초기화분포(Hypergeometric distribution)
모집단(N) 중에 부적합품 수가 M개이고 **비복원 추출**로 n개의 시료를 뽑았을 때, 그 중의 부적합품개수(불량품수) X는 X=x가 되는 확률 f(x)를 따른다.  
$$f(x) = \frac{_{M}\mathrm{C}_{x}*{}_{N-M}\mathrm{C}_{n-x}}{_{N}\mathrm{C}_{n}} = \frac{\begin{pmatrix} M  \\ x  \end{pmatrix} \begin{pmatrix} N-M  \\ n-x  \end{pmatrix}} {\begin{pmatrix} N  \\ n  \end{pmatrix}}$$

평균: <span>$$E(x) = n\frac{M}{N}$$</span>  
<p>$$f(x) = \frac{\begin{pmatrix} M  \\ x  \end{pmatrix} \begin{pmatrix} N-M  \\ n-x  \end{pmatrix}}{\begin{pmatrix} N  \\ n  \end{pmatrix}}$$</p>
<p>$$E(x) = \sum_{x=0}^{n} x\frac{\begin{pmatrix} M  \\ x  \end{pmatrix} \begin{pmatrix} N-M  \\ n-x  \end{pmatrix}}{\begin{pmatrix} N  \\ n  \end{pmatrix}}$$</p>
<p>$$= \sum_{x=1}^{n} x\frac{M!}{x!(M-x)!} \frac\begin{pmatrix} N-M  \\ n-x  \end{pmatrix}\begin{pmatrix} N  \\ n  \end{pmatrix}$$</p>
<p>$$= \sum_{x=1}^{n} M\frac{(M-1)!}{(x-1)!(M-x)!} \frac\begin{pmatrix} (N-1)-(M-1)  \\ (n-1)-(x-1)  \end{pmatrix}{ \frac{N}{n} \begin{pmatrix} (N-1)  \\ (n-1)  \end{pmatrix}} \because \begin{pmatrix} \alpha  \\ \beta  \end{pmatrix} = \frac{\alpha}{\beta}\begin{pmatrix} \alpha-1  \\ \beta-1  \end{pmatrix}$$</p>
위의 식에서 x-1 = X라고 치환하면 결과는 다음과 같다.  
<p>$$n\frac{M}{N} \sum_{X=0}^{n-1} \frac{ \begin{pmatrix} M-1  \\ x  \end{pmatrix} {(N-1) - (M-1) \choose (n-1) - X}}{\begin{pmatrix} N-1  \\ n-1  \end{pmatrix}}$$</p>
위의 식에서 <span>$$\sum_{X=0}^{n-1} \frac{ \begin{pmatrix} M-1  \\ x  \end{pmatrix} {(N-1) - (M-1) \choose (n-1) - X}}{\begin{pmatrix} N-1  \\ n-1  \end{pmatrix}}$$</span>는 초기화 분포의 모든 합 이므로 1이 되는 것을 확인할 수 있다.  
<p>$$\therefore E(x) = n\frac{M}{N}$$</p>
위의 최종적인 식에서 N은 전체 개수, M은 불량품의 갯수이다.  

**좀 더 생각하여 불량품을 뽑을 확률을 p라고 생각하면 다음과 같이 식을 변경할 수 있다.**  
<p>$$E(x) = np$$</p>
위의 식을 살펴보게 되면 이항분포에서의 기댓값과 같다는 것을 알 수 있다.  


분산: <span>$$V(x) = \frac{N-n}{N-1}n\frac{M}{N}(1-\frac{M}{N})$$</span>  
위의 식을 그대로 분산을 구하는 것은 많이 힘들기 때문에 편법을 사용하여 구한다.  
<p>$$V(x) = E(X(X-1))+E(x)-(E(x))^2$$</p>
<p>$$E(X(X-1)) = \sum_{x=0}^{n} x(x-1)\frac{\begin{pmatrix} M  \\ x  \end{pmatrix} \begin{pmatrix} N-M  \\ n-x  \end{pmatrix}}{\begin{pmatrix} N  \\ n  \end{pmatrix}}$$</p>
<p>$$= M(M-1) \sum_{x=2}^{n} \frac{\begin{pmatrix} M-2  \\ x-2  \end{pmatrix} \begin{pmatrix} (N-2)-(M-2)  \\ (n-2)-(x-2)  \end{pmatrix}}{\frac{N}{n} \frac{N-1}{n-1}\begin{pmatrix} N-2  \\ n-2  \end{pmatrix}}$$</p>
<p>$$= \frac{M(M-1)n(n-1)}{N(N-1)} \sum_{x=2}^{n} \frac{\begin{pmatrix} M-2  \\ x-2  \end{pmatrix} \begin{pmatrix} (N-2)-(M-2)  \\ (n-2)-(x-2)  \end{pmatrix}}{\begin{pmatrix} N-2  \\ n-2  \end{pmatrix}}$$</p>
평균에서와 똑같이 <span>$$\sum_{x=2}^{n} \frac{\begin{pmatrix} M-2  \\ x-2  \end{pmatrix} \begin{pmatrix} (N-2)-(M-2)  \\ (n-2)-(x-2)  \end{pmatrix}}{\begin{pmatrix} N-2  \\ n-2  \end{pmatrix}}$$</span>의 값이 적용되는 것을 알 수 있다.  
<p>$$\therefore V(x) = \frac{M(M-1)n(n-1)}{N(N-1)} + n\frac{M}{N} - (n\frac{M}{N})^2$$</p>
위의 식을 정리하면 다음과 같은 것을 알 수 있다.  
<p>$$V(x) = \frac{N-n}{N-1}n\frac{M}{N}(1-\frac{M}{N})$$</p>
**평균과 마찬가지로 불량품을 뽑을 확률을 p라고 생각하면 다음과 같이 식을 변경할 수 있다.**  
<p>$$V(x) = \frac{N-n}{N-1}p(1-p) = \frac{N-n}{N-1}pq$$</p>
ex) 1000개의 제품 중 13개가 풀량품이다. 1000개의 제품에서 임의로 20개를 뽑았을 때, 불량품이 3개가 포함될 확률을 구하시오.  
- N(전체 개수): 1000
- K(어떤 특성을 가진 제품의 수): 13
- n(표본수): 20
- x(표본에 포함된 어떤 특성의 제품 수): 3

<span>$$R_x=[0,1,2, ...., \alpha]$$</span>을 가능한 사상이라고 할 때 <span>$$\alpha$$</span>의 값은 다음과 같다.
<p>$$
\alpha=
\begin{cases}
k, & n \ge k \\
n, & n < k
\end{cases}
$$</p>
위의 조건을 통하여 문제를 풀어보면 식은 다음과 같다.  
<p>$$P(X=3)=\frac{\begin{pmatrix} K  \\ x  \end{pmatrix} \begin{pmatrix} N-K  \\ n-x  \end{pmatrix}}{\begin{pmatrix} N  \\ n  \end{pmatrix}} = \frac{\begin{pmatrix} 13  \\ 3  \end{pmatrix} \begin{pmatrix} 987  \\ 17  \end{pmatrix}}{\begin{pmatrix} 1000  \\ 20  \end{pmatrix}} \approx 0.00165$$</p>
**초기화 분포와 이항분포의 관계**  
기본적으로 **이항분포는 복원추출, 초기화분포는 비복원 추출이다.**  
이러한 이항분포와 초기화의 분포를 간단한 예를 통하여 확인하여 보자.  
ex) 예를 들어, 어떤 도시에 n명의 여자와 m명의 남자가 있습니다. 이 도시에 전염병이 퍼져 어떤 병을 가지고 있는 여자의 수를 X, 같은 전엽병을 가지오 있는 남자의 수를 Y라 할때, 전염병이 걸린 사람 중 여자일 확률을 구하려고 합니다. 여기서 남자와 여자는 전염병에 걸릴 확률이 p로 같습니다.  
여기서 X+Y = r이라고 한다면 Y=r-X가 되고 표로서 다음과 같이 나타낼 수 있다.  
<table>
    <tr>
        <td></td><td>여자</td><td>남자</td><td>계</td>
    </tr>
    <tr>
        <td>전염병O</td><td>x</td><td>r-x</td><td>r</td>
    </tr>
    <tr>
        <td>전염병x</td><td>n-x</td><td>m-r+x</td><td>n+m-r</td>
    </tr>
    <tr>
        <td>계</td><td>n</td><td>m</td><td>n+m</td>
    </tr>
</table>

여자의 수 X, 남자의 수 Y는 전염병에 걸릴 확률 p를 따르는 이항분포라고 가정하게 되면 두 확률 변수는 서로 독립이다.  
<p>$$X ~ (n,p), Y ~ (m,p)$$</p>
전염병에 걸린 사람 중에 여자일 확률을 구하게 되면 다음과 같다.  
<p>$$P(X=x|X+Y=r) = \frac{P(X=x \cap Y=r-x)}{P(X+Y=r)}$$</p>
X,Y는 서로 독립이므로 식을 다음과 같이 변형할 수 있다.  
<p>$$P(X=x|X+Y=r) = \frac{P(X=x)P(Y=r-x)}{P(X+Y=r)}$$</p>
각각의 확률은 이항분포를 따르므로 다음과 같이 정리될 수 있다.  
<p>$$P(X=x|X+Y=r) = \frac{_{m}\mathrm{C}_{r-x}p^rq^{m-r+x}{}_{n}\mathrm{C}_{x}p^xq^{n-x}}{_{n+m}\mathrm{C}_{r}p^rq^{n+m-r}}$$</p>
<p>$$=\frac{_{m}\mathrm{C}_{r-x}{}_{n}\mathrm{C}_{x}}{_{n+m}\mathrm{C}_{r}}$$</p>
위의 식을 살펴보면 **초기화 분포가 되는 것을 확인할 수 있다.**  

이와 반대로 초기화 분포에 극한 <span>$$(N \rightarrow \infty)$$</span>를 취하면 이항분포가 된다.  
조금 식을 변경하기 위하여 다음과 같은 Parameter를 정하고 f(x)를 구한뒤 계산해 보자.  
- N: 전체개수
- k: 불량품의 개수
- n: Sampling 개수
- x: 불량품 in Sampling
- p: 불량품을 뽑을 확률(= <span>$$\frac{k}{N}$$</span>)

위와 같은 Parameter가 존개하게 되면 초기화분포를 다음과 같이 나타낼 수 있다.  
<p>$$f(x) = \frac{_{k}\mathrm{C}_{x}{}_{N-k}\mathrm{C}_{n-x}}{_{N}\mathrm{C}_{n}}$$</p>
<p>$$= \frac{n!}{(n-x)!x!} * \frac{k!(N-K)!(N-n)!}{N!(k-x)!(N-k-n+x)!}$$</p>
<p>$$= {}_{n}\mathrm{C}_{x}*\frac{(N-n)!}{N!}*\frac{k!}{(k-x)!}*\frac{(N-k)!}{(N-k-n+x)!}$$</p>
<p>$$= {}_{n}\mathrm{C}_{x}*\frac{1}{N(N-1)...(N-n+1)}*(k(k-1)...(k-x+1))*((N-k)(N-k+1)...(N-k+n-x+1))$$</p>
위의 식을 <span>$$\frac{N^n}{N^n}$$</span>을 곱하게 되면 식은 다음과 같이 정리할 수 있다.  
<p>$$= {}_{n}\mathrm{C}_{x}*\frac{1}{1(1-\frac{1}{N})...(1-\frac{n+1}{N})}*(\frac{k}{N}(\frac{k}{N}-\frac{1}{N})...(\frac{k}{N}-\frac{x+1}{N}))*((1-\frac{k}{N})(1-\frac{k+1}{N})...(1-\frac{k+n-x+1}{N}))$$</p>
위의 식에서 <span>$$p=\frac{k}{N}$$</span>으로서 나타내고 <span>$$(N \rightarrow \infty)$$</span>으로서 값을 변경하면 다음과 같이 식을 나타낼 수 있다.<p>$$= {}_{n}\mathrm{C}_{x}*\frac{1}{1*1...1}*(p*...p)*((1-p)...(1-p))$$</p>
최종적으로 이항분포로 표현하기 위하여 <span>$$1-p = q$$</span>로서 나타내게 되면 정리된 식은 다음과 같다.  
<p>$${}_{n}\mathrm{C}_{x}p^x q^{n-x}$$</p>


따라서 둘의 관계를 표현하면 다음과 같이 나타낼 수 있다.  
<img src="https://mblogthumb-phinf.pstatic.net/20161019_66/mykepzzang_1476816211335trWYM_JPEG/%C7%C1%B7%B9%C1%A8%C5%D7%C0%CC%BC%C71.jpg?type=w2"/><br>
사진 출처: <a href="https://m.blog.naver.com/mykepzzang/220839789774">mykepzzang 블로그</a><br>

#### (4) 포아송분포(Poisson distribution)
확률변수 X를 시간 (0,t)에서 발생하는 사건의 수라 하면 확률함수 f(x)는 다음과 같다.  
<p>$$f(x) = \frac{e^{-\lambda t}(\lambda t)^x}{x!}$$</p>
(단, <span>$$\lambda$$</span>= 단위 시간당 평균 발생 건수(모수), <span>$$\lambda > 0, x=0,1,2,3,... $$</span>, e=2.71828... 이다.)  

평균: <span>$$E(x) = \lambda$$</span>  
<p>$$E(x) = \sum_{x=0}^{\infty} xf(x) = \sum_{x=0}^{\infty} x\frac{e^{-\lambda t}(\lambda t)^x}{x!}$$</p>
<p>$$= \lambda \sum_{x=1}^{\infty} \frac{e^{-\lambda t}(\lambda t)^{x-1}}{(x-1)!}$$</p>
위의 식에서 <span>$$\sum_{x=1}^{\infty} \frac{e^{-\lambda t}(\lambda t)^{x-1}}{(x-1)!}$$</span>은 f(x-1)의 확률의 총합 이므로  
<p>$$\therefore E(x) = \lambda$$</p>
분산: <span>$$V(x) = \lambda$$</span>  
<p>$$V(x) = E(x(x-1))+E(x)-(E(x))^2 = E(x(x-1))+ \lambda-\lambda^2$$</p>
<p>$$E(x(x-1)) = \sum_{x=0}^{\infty} x(x-1)f(x) = \sum_{x=0}^{\infty} x(x-1)\frac{e^{-\lambda t}(\lambda t)^x}{x!}$$</p>
<p>$$= \lambda^2 \sum_{x=2}^{\infty} \frac{e^{-\lambda t}(\lambda t)^{x-2}}{(x-2)!}$$</p>
위의 식에서 <span>$$\sum_{x=2}^{\infty} \frac{e^{-\lambda t}(\lambda t)^{x-2}}{(x-2)!}$$</span>은 f(x-2)의 확률의 총합 이므로  
<p>$$\therefore E(x(x-1)) = \lambda^2$$</p>
<p>$$\therefore V(x) = \lambda^2 + \lambda -\lambda^2 = \lambda$$</p>
**포아송 프로세스**  
포아송 프로세스란 어떤 사건의 발생횟수가 포아송 분포를 따르는 확률과정이다.  
즉, 포아송 분포를 따르기 위하여 필요한 조건이라고 생각할 수 있다.  
(참고, 사건(Event)=시간과 관계가 없다, 프로세스(process)=시간에 따라 확률이 변할 수 있다.)  
1. 독립성: 어떤 단위 시간 또는 공간에서 발생한 결과는 중복되지 않은 다른 시간이나 공간에서 발생한 결과와 서로 독립이다.
2. 일정성: 어떤 단위 시간 또는 단위 공간에서 발생한 확률(또는 횟수)은 그 시간의 크기, 혹은 공간의 크기에 비례하고, 외부의 영향을 받지 않는다. 즉 단위 시간이나 공간에서 발생한 평균발생 횟수는 일정하다.  
3. 비집락성: 매우 짧은 시간이나 매우 작은 공간에 두 개 이상의 결과가 동시에 발생할 확률은 0이다.

위의 3가지 포아송 프로세스에 관하여 <a href="https://m.blog.naver.com/mykepzzang/220840724901">mykepzzang 블로그</a>에서 좋은 예시를 보여주었다.  
>1. 9시부터 10시까지 우리은행에 들른 고객의 수와 같은 시간 동안 국민은행에 들른 고객의 수는 서로 독립이고, 또한 9시부터 10시까지 우리은행에 들른 고객의 수와 10시부터 11시까지 신한은행에 들른 고객의 수도 역시 서로 독립이다.
2. 만약 어떤 사건이 1분에 평균 2번 발생한다면, 3분동안에는 평균 6번이 발생한다.
3. 서해안고속도로에 같은 시간에 같은 지점에서 같은 교통사고가 두 번 이상 발생할 확률은 무시해도 좋다.
>

ex) 시간당 평균 방문자수가 10명인 음식점에 오전 9~10시에 12명이 방문할 확률을 구하여라.  
<p>$$P(X=12) = \frac{e^{-10}(10)^12}{12!} \approx 0.0948$$</p>
위와 같은 포아송의 경우 계산이 복잡하기 때문에 포아송 분포표를 보고 계산하는 것이 빠르다.  

**포아송분포와 이항분포의 관계**  
**포아송분포의 조건부 분포는 이항분포 이다.**  
위의 말을 증명하기 위해서 다음과 같은 가정을 하고 식을 유도하여 보자.  
<p>$$X \text{~} Pois(\lambda_1), Y \text{~} Pois(\lambda_2)$$</p>
X,Y가 서로 독립이므로(포아송 프로세스 1번째 독립성) 다음과 같은 식을 사용할 수 있다.  
<p>$$X+Y \text{~} Pois(\lambda_1+\lambda_2)$$</p>
<p>$$P(X=x|X+Y=n) = \frac{P(X=x \cap X+Y=n)}{P(X+Y=n)}$$</p>
위의 식에서 X,Y가 서로 독립이므로 다음의 식이 성립한다.  
<p>$$P(X=x|X+Y=n) = \frac{P(X=x)P(Y=n-X)}{P(X+Y=n)}$$</p>
<p>$$=\frac{ \frac{e^{-\lambda_1}\lambda_1^x}{x!}  \frac{e^{-\lambda_1}\lambda_1^y}{y!} }{\frac{e^{-(\lambda_1+\lambda_2)}(\lambda_1+\lambda_2)^n}{n!}}$$</p>
<p>$$=\frac{n!}{(n-x)!x!}*\frac{(\lambda_1^x * \lambda_2^{n-x})}{(\lambda_1+\lambda_2)^n}$$</p>
<p>$$= {}_{n}\mathrm{C}_{x}*(\frac{\lambda_1}{\lambda_1+\lambda_2})*(\frac{\lambda_2}{\lambda_1+\lambda_2})$$</p>
위의 식에서 <span>$$\frac{\lambda_1}{\lambda_1+\lambda_2} = p$$</span>라고 치환하면 <span>$$\frac{\lambda_2}{\lambda_1+\lambda_2} = 1-p = q$$</span>이다.  
<p>$$\therefore P(X=x|X+Y=n) = {}_{n}\mathrm{C}_{x} p^x q^{n-x}$$</p>
위의 식을 살펴보면 이항분포인 것을 확인할 수 있다.  

**이항분포의 극한은 포아송분포 이다.**  
위의 말을 증명하기 위해서 다음과 같은 가정을 하고 식을 유도하여 보자.  
먼저 포아송분포에서 <span>$$\lambda$$</span>를 모수이자 평균으로서 정의하였다.  
이것을 이항분포의 평균에 대입하면 <span>$$\lambda = np \rightarrow p = \frac{\lambda}{n}$$</span>이 성립한다.  
<p>$$P(X=x) = {}_{n}\mathrm{C}_{x} p^x (1-p)^{n-x}$$</p>
<p>$$= \frac{\lambda^x}{x!}*\frac{n(n-1)...(n-x+1)}{n^x}*(1-\frac{\lambda}{n})^{n}*(1-\frac{\lambda}{n})^{-x}$$</p>
<p>$$= \frac{\lambda^x}{x!}*(1(1-\frac{1}{n})...(1-\frac{x+1}{n}))*(1-\frac{\lambda}{n})^{-x}$$</p>
만약 <span>$$n \rightarrow \infty$$</span>가 되면 각각은 다음과 같이 변하게 된다.  
<p>$$(1(1-\frac{1}{n})...(1-\frac{x+1}{n})) \rightarrow 1$$</p>
<p>$$(1-\frac{x+1}{n}) \rightarrow 1$$</p>
<p>$$((1-\frac{\lambda}{n})^{-x} \rightarrow 1$$</p>
<p>$$P(X=x) = {}_{n}\mathrm{C}_{x} p^x (1-p)^{n-x}$$</p>
따라서 둘의 관계를 표현하면 다음과 같이 나타낼 수 있다.  
<img src="https://mblogthumb-phinf.pstatic.net/MjAxNjEwMjBfMTI0/MDAxNDc2OTA1MTEzMjY2.ZdUCNP0gw9X5nhm2dDapcF6EKfArTlysvixz2xn1j4Qg.ICh2sHOBQ8gQkpnBT2KJYHZJc7ZLC13ppxDCQsjzwP0g.JPEG.mykepzzang/%ED%94%84%EB%A0%88%EC%A0%A0%ED%85%8C%EC%9D%B4%EC%85%981.jpg?type=w2"/><br>
사진 출처: <a href="https://m.blog.naver.com/mykepzzang/220840724901">mykepzzang 블로그</a><br>

<hr>
참조: <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업</a><br>
참조: <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


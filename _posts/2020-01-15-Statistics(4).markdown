---
layout: post
title:  "Statistics(4)-Sampling distribution"
date:   2020-01-20 09:30:20 +0700
categories: [statistics]
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
이번 POST는 <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업 내용</a>을 정리한 것 입니다.  
문제나 자세한 내용은 <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a>를 참조하였습니다.  

### 6. 샘플링 분포
모집단(Population)은 연구의 대상이 되는 모든 개체들의 집합이다.  
개체들의 특성을 나타내는 것을 확률변수 X라 한다면, 모집단의 분포는 <span>$X \text{~} N(\mu,\sigma^2)$</span>(모집단의 크기가 충분히 크다면 분포는 정규분포를 따를 것 이다.)로 나타낼 수 있다.  
**이때 <span>$\mu, \sigma^2$</span>을 모수라고 한다.**  
모수는 모집단의 분포를 결정하는 상수로서 대체로 알려져 있지 않다.  
모집단의 크기는 매우 크다고 가정을 하고 있고, 이러한 모든 Sample을 계산할 수 없기 때문이다.  
따라서 이러한 **모딥단으로부터 모수를 연구하기 위하여 소수의 개체들을 추출하는 과정을 샘플링(Sampling)이라 한다.** 이 때 추출된 소수의 개체들은 모집단을 연구하는데 사용되는중요한 자료이므로 모집단을 잘 나타낼 수 있도록 해야 하며, 독립적이어야 한다.  
**이렇게 추출된 소수의 개체들의 집합을 표본(Sample)이라고 한다.**  
표본 <span>$X_1, X_2, ..., X_n$</span>은 서로 독립적이면서 모집단의 분포와 동일한 분포를 갖게 된다.  
**이러한 표본의 특성을 기호로 나타내면 <span>$X_i \text{~} iid N(\mu,\sigma^2)$</span>이며 각각의 iid의 의미는 다음과 같다.**  

- i(independent): 독립성(Sample의 각각은 서로 독립이다.)
- i(identical): 동일성(모집단과 동일한 특성(<span>$$\mu, \sigma^2$$</span>)을 가지고 있다.)
- d(distribution): 분포

**통계량(Statistic)은 표본 데이터 <span>$$X_1, X_2, ..., X_n$$</span>(Population에서 n개의 Data를 Sampling한다.)의 함수이며 확률변수이다.(Sample을 어떻게 정하냐에 따라서 값이 변함으로 값이 정해져있는 상수가 아닌 값이 변할 수 있는 통계량이라고 지칭한다.)**  
통계량의 예로서 대표적인 것이 표본평균과 표본분산이다.  
이러한 통계량을 통하여 알 수 없었던 모수(Parameter: <span>$$\theta(\mu, \sigma^2)$$</span>)을 알아내는 것이 목표이다.  

#### (1) 표본평균(Sample mean)의 분포
표본평균은 <span>$$\bar{X}$$</span>로서 표현하고 다음과 같은 식으로서 나타낸다.  
<p>$$\bar{X} = \frac{\sum_{i-1}^{n}X_i}{n}$$</p>
위의 식에서 <span>$$X_i \text{~} iid N(\mu,\sigma^2)$$</span>이므로 선형결합된 <span>$$\bar{X}$$</span>또한 정규분포를 따를 것이라는 것은 생각할 수 있다.  

평균: <span>$$E(\bar{X}) = \mu$$</span>  
<p>$$E(\bar{X}) = E(\frac{1}{n}(X_1+X_2+...+X_n)) = \frac{1}{n}(E(X_1)+E(X_2)+...+E(X_n))$$</p>
<p>$$=\frac{1}{n}(\mu+\mu+...+\mu)=\mu$$</p>
분산: <span>$$V(\bar{X}) = \frac{\sigma^2}{n}$$</span>  
<p>$$V(\bar{X}) = V(\frac{1}{n}(X_1+X_2+...+X_n)) = \frac{1}{n^2}(V(X_1)+V(X_2)+...+V(X_n))$$</p>
<p>$$=\frac{1}{n^2}(\sigma^2+\sigma^2+...+\sigma^2)=\frac{\sigma^2}{n}$$</p>
위에서 평균과 분산과 정규분포이 특성을 생각하면 다음과 같이 나타낼 수 있다.  
<p>$$\bar{X} = N(\mu,\frac{\sigma^2}{n})$$</p>
#### (2) 카이제곱분포(Chi-square distribution)
표본분산의 분포를 구하기 전에 카이제곱의 분포를 알아야 한다.  
만약 확률변수 Z가 정규분포를 따른다고 가정한다면 새로운 변수 <span>$$Y=Z^2$$</span>는 자유도가 1인 카이제곱분포를 따른다고 한다. 먄약, <span>$$Z_i \text{~} iid N(0,1)$$</span>라면 <span>$$V = Z_1^2+Z_2^2+...+Z_n^2$$</span>은 자유도가 n인 카이제곱분포를 따른다고 하며 기호로서는 <span>$$V \text{~} \chi_{(n)}^2$$</span>로서 표현한다.  

평균: <span>$$E(X) = n$$</span>  
분산: <span>$$V(X) = 2n$$</span>  

**카이제곱분포의 특징**  
만약 각각의 자유도가 <span>$$u \text{~} \chi_{(n)}^2, v \text{~} \chi_{(k)}^2$$</span>이라고 할 때 각각을 만족한다.  
1. <span>$$u+v = \chi_{(n+k)}^2$$</span>
2. <span>$$u-v = \chi_{(n-k)}^2, \text{  if }n>k$$</span>

카이제곱의 더 자세한 내용이나 평균, 분산에 대한 증명은 아래 링크를 참조하자.  
참조: <a href="https://blog.naver.com/PostView.nhn?blogId=mykepzzang&logNo=220852102307&parentCategoryNo=&categoryNo=&viewDate=&isShowPopularPosts=false&from=postView">카이제곱분포 자세한 내용</a><br>

#### (3) 표본분산(Sample variance)의 분포
표본분산은 <span>$$S^2$$</span>로서 표현하고 다음과 같은 식으로서 나타낸다.  
<p>$$S^2 = \frac{\sum_{i-1}^{n}(X_i-\bar{X})^2}{n-1}$$</p>
위의 식에서 <span>$$\frac{\sum_{i-1}^{n}(X_i-\bar{X})^2}{n-1}$$</span>은 제곱이 포함되어 있어서 선형결합이라고 표현할 수 없으로 그로인하여 카이제곱으로서 식을 유도하게 된다. **기본적은 우리가 배워왔던 n으로서 나누는 것이 아닌 n-1로 나누는 것 또한 식을 유도하면 이해하게 될 것이다.**    

평균: <span>$$E(S^2) = \sigma^2$$</span>  
분산: <span>$$V(S^2) = 2\sigma^2$$</span>  

각각의 평균과 분산을 나타내는 것을 구하는 것이 아닌 카이제곱형태로서 변형하여서 쉽게 구하는 방식으로서 식을 유도하여 보자.  
<p>$$\sum(X_i -\bar{X})^2 = \sum(X_i - \bar{X} + \bar{X} - \mu)^2$$</p>
<p>$$= \sum(X_i -\bar{X})^2 + \sum(\bar{X} -\mu)^2 + 2\sum(X_i -\bar{X})(\bar{X} -\mu)$$</p>
<p>$$= \sum(X_i -\bar{X})^2 + \sum(\bar{X} -\mu)^2 \because \sum(X_i -\mu)\text{는 편차의 합이기 떄문이다.}$$</p>
<p>$$ \rightarrow \sum(\frac{X_i -\bar{X}}{\sigma})^2 = \sum(\frac{X_i -\bar{X}}{\sigma})^2 + \sum(\frac{\bar{X} -\mu}{\sigma})^2$$</p>
각각의 합(<span>$$\sum(\frac{X_i -\bar{X}}{\sigma})^2$$</span>, <span>$$\sum(\frac{\bar{X} -\mu}{\sigma})^2$$</span>)을 Z변환하여 정규화한 하고 카이제곱을 적용시키면 다음과 같다.  
<p>$$\sum(\frac{X_i -\bar{X}}{\sigma})^2 = \sum_{i=1}^{n} Z_i^2 = \chi_{n}^2$$</p>
<p>$$\sum(\frac{\bar{X} -\mu}{\sigma})^2 = n(\frac{\bar{X}-\mu}{\sigma})^2=\frac{\bar{X}-\mu}{\frac{\sigma}{\sqrt{n}}})^2 = \chi_{1}^2 \because \bar{X} \text{~} N(\mu,\frac{\sigma^2}{n})$$</p>
<p>$$\therefore \chi_{n}^2 = \sum(\frac{X_i -\bar{X}}{\sigma})^2 + \chi_{1}^2 \rightarrow \sum(\frac{X_i -\bar{X}}{\sigma})^2 = \chi_{(n-1)}^2$$</p>
위의 식을 다시 표본분산에 맞게 정리하면 다음과 같다.  
<p>$$S^2 = \frac{\sum_{i-1}^{n}(X_i-\bar{X})^2}{n-1} \rightarrow \frac{(n-1)S^2}{\sigma^2} \text{~} \chi^2_{(n-1)}$$</p>
위의 식에서 Chisquare의 각각의 평균과 분산은 n-1, 2(n-1)이므로 최종적인 평균과 분산은 다음과 같다.  
<p>$$E(S^2) = (n-1)\frac{\sigma^2}{(n-1)} = \sigma^2 \text{ ,  } V(S^2) = 2(n-1)\frac{\sigma^2}{(n-1)} = 2\sigma^2$$</p>
**위에서 표본평균 분포의 기댓값과 표본분산 분포의 기댓값을 활용하여 구할 수 없었던 모수의 평균과 분산을 추정할 수 있다.**  

<hr>
참조: <a href="http://www.kocw.net/home/cview.do?mty=p&kemId=1165032">한양대학교 수리통계학 수업</a><br>
참조: <a href="https://m.blog.naver.com/mykepzzang/220838509912">mykepzzang 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


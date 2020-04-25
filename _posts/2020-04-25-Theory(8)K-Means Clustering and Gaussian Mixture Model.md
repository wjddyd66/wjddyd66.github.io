---
layout: post
title:  "Theory8. K-Means Clustering and Gaussian Mixture Model"
date:   2020-04-25 10:55:20 +0700
categories: [Machine Learning]
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## 8. K-Means Clustering and Gaussian Mixture Model
$$\newcommand{\argmin}{\mathop{\mathrm{argmin}}\limits}$$
$$\newcommand{\argmax}{\mathop{\mathrm{argmax}}\limits}$$
Machine Learning의 기초적인 이론부분을 다시 제대로 잡고 싶어서 <a href="https://kaist.edwith.org/machinelearning2__17/joinLectures/9782">문일철 교수님의 인공지능 및 기계학습 개론</a>을 정리한 Post입니다.

- 8.1 K-Means Algorithm
- 8.2 Hierarchical Algorithm

### 8.1 K-Means Algorithm

K-Means Clusturing은 분리형 군집화 알고리즘중 하나로서, **각 군집은 하나의 중심을 가지고 각 개체는 가장 가까운 중심에 할당되며, 같은 중심에 개체들이 모여 하나의 군집을 형성하는 방법이다.**  
이러한 방식은 사전에 군집 수(k)를 사용자가 지정해야하는 hyperparameter이다.  

<p>$$W(C) = \frac{1}{2} \sum_{k=1}^{K} \sum_{C(i)=C(j)=k} d_E (m_i,x_j)^2$$</p>
<p>$$d_{E}(a,b) = \sqrt{a^2+b^2}$$</p>

=> 같은 Cluster내에 있는 DataPoint끼리의 Euclidian Distance의 합이 최소가 되도록 Centeriod를 Update한다.

**학습 과정**  
**1. 클러스터 개수 K를 고정하고, t=0으로 초기화 한다. K 개의 클러스터( <span>$$C^{0}_{i}, i=1, ..., K$$</span>)의 평균(<span>$$m^{0}_{i}, i=1, ..., K$$</span>)을 임의로 선택한다.**  
<img src="http://i.imgur.com/hgNzXsc.png" width="500px" title="source: imgur.com"><br>

**2. 클러스터링하려는 데이터(<span>$$x_{j}, j=1, ..., M$$</span>)각각의 K개의 클러스터 평균과의 최소거리가 되는 클러스터(<span>$$C^{t}_{p}$$</span>)로 <span>$$x_{j}$$</span>를 분류**  
<img src="http://i.imgur.com/OFn22dM.png" width="500px" title="source: imgur.com"><br>

**3. 각 클러스터(<span>$$C^{t}_{i}, i=1, ..., K$$</span>)에 속한 데이터를 이용하여 새로운 클러스터 평균 <span>$$m^{t+1}_{i}, i=1, ..., K$$</span> 을 계산한다.**  
<p>$m^{t+1}_{i} = \frac{1}{C^{t}_{i}} \sum_{x \in C^{t}_{i}}x_{j}, i=1, ..., K$</p>
<img src="http://i.imgur.com/UmBgHhf.png" width="500px" title="source: imgur.com"><br>

**4. t = t+1로 증가시키고 최대 반복회수까지 2~3단계를 계속하여 반복한다.**  
**2단계 반복**  
<img src="http://i.imgur.com/DWmbUxP.png" width="500px" title="source: imgur.com"><br>
**3단계 반복**  
<img src="http://i.imgur.com/OBhtsbV.png" width="500px" title="source: imgur.com">
<br>

**장점 및 단점**  
장점: 
 - 계산 복잡성이 O(n)이여서 가벼운 편이다.
 - 알고리즘이 쉽다.

단점:  
 - 군집의 크기가 다를 경우 제대로 작동하지 않을 수 있다.
<img src="http://i.imgur.com/IH8FAfq.png" width="500px" title="source: imgur.com">
 - 군집의 밀도가 다를 경우 제대로 작동하지 않을 수 있다.
<img src="http://i.imgur.com/pJmhpQ6.png" width="500px" title="source: imgur.com">
 - 데이터 분포가 특이한 케이스인 경우 제대로 작동하지 않을 수 있다.
<img src="http://i.imgur.com/v37p0Gi.png" width="500px" title="source: imgur.com">

- 군집 수(k)를 사용자가 미리 지정해야 한다.
- Outlier에 Robust하지 않다.(K-Medoids를 사용하면 해결 가능하나 Complexity가 증가되어서 사용되지 않는다.)
- Hard Clustering 이다. Cluster에 속하냐 안하냐로서만 표현하지, Probability로서 표현되지는 않는다.

그림 출처: <a href="https://ratsgo.github.io/machine%20learning/2017/04/19/KC/">ratsgo Blog</a>
<br>

**참조**  
1. K-Means Algorithm은 EM Algorithm이다. Cluster에 속하는지에 대한 <span>$$C$$</span>를 Expected -> Cluster의 평균 Centriod(<span>$$m_i$$</span>)를 Update한다는 의미이다.
2. EM Algorithm이므로 Local Minimum에 빠질 가능성이 있다. 따라서 Centriod의 위치를 바꿔가면서 수행하여야 한다.

### 8.2 Hierarchical Algorithm
**K-Means와 달리 K개의 Cluster의 개수를 지정하지 않고 Dendrogram Structure로서 Cluster하는 방식이다.**  

장점:
 - Cluster의 개수를 지정하지 않는다.
 - Cluster끼리의 관계 파악 가능

단점:
 - Post-Processing: 결국 Dendrogram Structure에서 Cluster로서 나누기 위해 Cut하여 나누어야 됨(사용자가 직접 지정 or Dynamic Cut(R) 패키지 사용)
 - 하나의 Cluster로서 결정되면 바뀌지 않는다.
 - Complexity가 K-Means(O(n))에 비하여 O(n^3)로서 무거운 편 이다.

Hierarchical Clustering은 크게 2가지로서 나누어 진다.  
- Agglomerative Hierarchical Clustering: Bottom-up Algorithm
- Divisive Hierarchical Clustering: Top-Down Algorithm

위와 같이 크게 2가지의 Hierarchical Clustering방법에서 Cluster끼리의 거리를 계산하기 위하여 공통적으로 사용하는 방법은 다음과 같다.  

Cluster Distance 측정 방법(위의 2가지 방법 둘다 공통으로 사용)  
- Single Linkage: Cluster안에서의 Point끼리 가장 가까운 거리를 측정(Smallest Distance)
- Complete Linkage: Cluster안에서의 Point끼리 가장 먼 거리를 측정(Largest Distance)
- Centroid Linkage: Cluster안에서의 Point끼리 중심 거리를 측정(Distance Between two Centroids)
- Average Linkage: Cluster안에서의 Point끼리 거리의 평균을 측정(Average Distance)

<img src="http://i.imgur.com/TM1PuQc.png" width="600px" title="source: imgur.com">

**학습 방법**  
Bottom Up 방식(Agglomerative Hierarchical Clustering)으로 예시를 들면 다음과 같다.  

1. Cluster끼리의 거리 계산(Euclidian Distance or Correlation Distance, 위의 4가지 방법 중 하나를 선택)를 계산하여 Matrix형태로서 나타낸다.  
<img src="http://i.imgur.com/25IT5fI.png" width="500px" title="source: imgur.com"><br>

2. 가장 가까운 거리의 Point들을 하나의 Cluster로서 지정한다.
<img src="http://i.imgur.com/iuZm5wl.png" width="500px" title="source: imgur.com">

3. 다시 Cluster의 거리를 끼리 거기를 계산하고 1~2의 방법을 계속하여 수행한다.
<img src="http://i.imgur.com/1OI9j9S.png" width="500px" title="source: imgur.com">

4. 최종적으로 Cluster가 1개가 되었을 때(Root Cluster를 구하였을 때) 학습을 종료한다.
<img src="http://i.imgur.com/S18WtII.png" width="500px" title="source: imgur.com">

5. 위와 같은 최종적인 Dendrogram Structure에서 사용자가 어떤 Cluster로 분류할 지 정하여 Cut(Post-Processing) 한다. Ex) 만약 두번째 층에서 끝게 되면 -> (A,D), C, B로서 3개의 Cluster로서 분할이 가능하다.

그림 출처: <a href="https://ratsgo.github.io/machine%20learning/2017/04/18/HC/">ratsgo 블로그</a>

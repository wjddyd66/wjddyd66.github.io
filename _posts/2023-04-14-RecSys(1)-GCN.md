---
layout: post
title:  "GCN"
date:   2023-04-14 11:00:20 +0700
categories: [RecSys]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['<span><span>$$','<span><span>$$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS

### 참조 링크

- <a href="https://arxiv.org/pdf/1609.02907.pdf">SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS</a> (Graph Convolution Network Paper)
- <a href="https://arxiv.org/pdf/1606.09375.pdf">Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering</a> (ChebNet Paper)
- <a href="https://velog.io/@bailando/MLDL-Inductive-learning-과-Transductive-learning">bailando Blog</a> (About Trunsductive)
- <a href="https://thejb.ai/gcn/">thejb Blog</a> (GCN의 의미)
- <a href="https://yamalab.tistory.com/164">yamalab Blog</a> (Graph에서 Convolution 이란)
- <a href="https://ralasun.github.io/deep%20learning/2021/02/15/gcn/">ralasun Blog</a> (Spectral Graph Convolution 이란)
- <a href="https://darkpgmr.tistory.com/171">다크프로그래머 Blog</a> (Fourier Transform 이란)
- <a href="https://wjddyd66.github.io/others/PCA/">wjddyd66 Blog</a> (Symmetric Matrix 특징과 Eigen-Decomposition)
- <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a> (Laplacian 이란)
- <a href="https://en.wikipedia.org/wiki/Chebyshev_polynomials">WikiPedia</a> (Chebyshev_polynomials)
- <a href="https://www.youtube.com/watch?v=8YNdlXtMn40&t=184s">YouTube</a> (GCN 종류 및 요약 #1)
- <a href="https://www.youtube.com/watch?v=y52qSiGOhbs">YouTube</a> (GCN 종류 및 요약 #2)
- <a href="https://www.youtube.com/watch?v=w4YDI715DG0&t=1802s">YouTube</a> (GCN 종류 및 요약 #3)

### Abstract
>We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.

해당 논문에서는 **GCN(Graph Convolution Network)** 를 제안한다. 개인적으로 생각하는 GCN은 아래와 같은 특징을 가지고 있다.
1. **<a href="https://velog.io/@bailando/MLDL-Inductive-learning-과-Transductive-learning">Trunsductive</a> 접근 법으로서 Label이 없는 Dataset도 활용하여 학습한다. (Semi-supervised classification) 이러한 방법은, 많은 Node에 Label이 없는 Graph구조의 Dataset에 적합하다.**
2. **first-order approximation of spectral graph convolution을 제시한다. -> Convolution Neural Network로서 주변 노드의 정보를 사용할 수 있는 Model을 제안한다.**

### Introduction
**Problem**: 제시되는 방법이 나오기 전에 Graph Laplacian regularization term (semi-supervised)을 살펴보면 아래와 같다.

<p>$$L = L_0 + \lambda L_{reg}, \text{with } L_{reg} = \sum_{i,j} A_{i,j} \| f(X_i) - f(X_j)\|^2 - (1)$$</p>

- <span>$$\lambda$$</span>: weight (hyper-parameter)
- <span>$$X$$</span>: Matrix of node feature vectors
- <span>$$X_i$$</span>: i-th Node feature vector
- <span>$$A \in \mathbb{R}^{N \times N}$$</span>: adjacency matrix

위의 수식을 살펴보면, Label이 있는 Node에 대하여서는 Supervised Loss를 적용할 수 있다. **하지만, Label이 없는 경우에는 <span>$$A_{i,j} \| f(X_i) - f(X_j)\|^2$$</span>을 통하여 이여져있는 노드들은 같은 Label을 가지게 된다. (<span>$$A_{i,j}$$</span>는 Binary or weighted 이나 생각하기 편하게 Binary라 생각한다면)**

이러한 수식의 문제점은 Graph로서의 수용력 매우 제한하는 문제를 발생시킨다. 해당 논문에서는 다음과 같이 표현하였다. "This assumption, however, might restrict modeling capacity, as graph edges need not necessarily encode node similarity, but could contain additional information."

**Solution**: 해당 논문에서는 adjacency matrix (<span>$$A$$</span>)와 Matrix of node feature vectors (<span>$$X$$</span>)를 활용하는 <span>$$f(\cdot)$$</span>을 함수를 제안한다. 또한, <span>$$f(\cdot)$$</span>를 활용하게 되면, 위의 식 (1)에서 Label이 있는 Node만 사용하여 문제점으로 제시한 "주변 연결된 Node들은 같은 Label 을 가진다"라는 문제점을 해결한다.

### FAST APPROXIMATE CONVOLUTIONS ON GRAPHS
제안하는 <span>$$f(A, X)$$</span> (Graph Convolutional Network (GCN))은 아래와 같다.

<p>$$H^{(l+1)} = \sigma (\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) - (2)$$</p>
- <span>$$\tilde{A} = A + I_N$$</span>
- <span>$$\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$$</span>
- <span>$$W^{(l)}$$</span>: layer-specific trainable weight matrix.
- <span>$$\sigma (\cdot)$$</span>: Activation function (ex) ReLU)
- <span>$$H^{(0)} = X$$</span>

위에서 정의한 식은 일반적으로 알고 있는 Convolution Neural Network와 많이 다르다. 일반적으로 Image에서 사용하는 Convolution과 Graph에서 사용하는 Convolution이 달라야 하는 이유는 아래 Figure를 참조하자.
![png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZGepm%2FbtrapF7WmRg%2FN8WmTqUqdxaPCcRkZujIok%2Fimg.png)<br>
그림 참조: <a href="https://yamalab.tistory.com/164">yamalab Blog</a>

**위의 그림을 참조하면, GCN의 Convolution은 Graph의 Adjency Matrix (<span>$$A$$</span>)의 순서가 달라져도 고려할 수 있어야 한다. 즉, 위치 정보와 상관 없이 같은 값으로서 Output을 내보내야 한다. 그러한 식은 위의 식 (2) 이며 하나한 의미를 알아가보자.**

### SPECTRAL GRAPH CONVOLUTIONS

**해당 Section은 <a href="https://ralasun.github.io/deep%20learning/2021/02/15/gcn/">ralasun Blog</a>의 내용을 이해한 대로 정리한 Section입니다. 정확한 수식과 내용은 해당 Blog를 참조해 주세요.**

위에서 Graph에서 Convolution Filter를 사용하려면 (1) Local invariance (그래프의 구조에 따라 Output 값이 변하면 안됨. Relation을 고려해야 함), (2) Filter의 크기는 얼만큼의 Node를 포함하여야 하는지 표현해야한다. 라는 2가지 조건을 만족하여야 한다. 그림으로 표현하면 아래와 같다.

**Figure 1**  
<img src="https://imgur.com/KWmqbgk.png" width="200">
그림 참조: <a href="https://ralasun.github.io/deep%20learning/2021/02/15/gcn/">ralasun Blog</a>

**Graph Convolution은 위와 같이 2가지 조건을 만족하는 방법을 Spectral Graph Convolution을 제안하였다.**

먼저, 해당 방법을 이해하기 위해서는 "Message Passing"을 이해하여야 한다. Graph에서는 Adjacency matrix (<span>$$A$$</span>)를 사용하기 때문에 아래와 같은 특성을 가지게 됩니다.

**Figure 2**  
<img src="https://imgur.com/Fv2FJbC.png" width="800">
그림 참조: <a href="https://ralasun.github.io/deep%20learning/2021/02/15/gcn/">ralasun Blog</a>

위의 그림을 해석하게 되면, 시간 (t)이 지남에 따라 (Graph Neural Network에서 Layer를 쌓으면서 Adjacency matrix를 곱함)에 따라서, 자기 자신의 신호 뿐만 아니라, 주변 신호를 사용한다는 것 이다. 

Figure 1과 Figure 2에 대해서 함께 생각해보자. Figure 2에서 시간 (t)이 지난 다는 것은 Figure 1에서 얼만 큼 떨어진 Node (k)를 사용할 지 정의할 수 있다. 즉 조건 (2)를 만족한다. 또한, Figure 2에서 시간 (t)가 지남에 따라 각 Node에 어떠한 값들이 있는지 생각해보면, 자기 자신의 신호 + 주변 값들의 신호 (Aggregeation은 Summation을 사용하면) 로서 이루워 진다. 즉, 그래프의 Relation을 고려한 결과 (조건 (1) 만족)이지 그래프의 구조에 따라 변하지 않는 것을 확인할 수 있다.

해당 조건을 모두 만족하는 그림을 단순화 하면 Figure 2의 왼쪽 그림으로 생각할 수 있다. -> **Spectral graph convolution은 이러한 신호를 분석하기 위하여 Signal Processing에서 사용하는 Fourier Transform을 사용한다는 것 이다.**

### Fourier Transform
**해당 Section은 간략한 내용에 대해서만 설명합니다. (Fourier Transform에 대해서 세세히 설명하기에는 너무 방대한 양 입니다.)정확한 수식과 내용은 <a href="https://darkpgmr.tistory.com/171">다크프로그래머 Blog</a>를 참조해 주세요.**

Fourier Transform은 Time/spatial domain -> Frequence Domain으로 옮기는 과정이다. 아래 그림을 살펴보면 다음과 같다.

![png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F9967FA3359B63D8122)

해당 그림을 해석하면, 빨간색 신호는 다양한 파란색 주기함수로서 이루워져있다. 파란색 주기함수는 각자의 고유한 주파수(frequency)와 강도(amplitude)를 가지고 있다.

퓨리에 변환의 식은 아래와 같다.
<p>$$f(x) = \int_{-\infty}^{\infty} F(u) e^{j 2\pi u x}du$$</p>
<p>$$F(u) = \int_{-\infty}^{\infty} f(x) e^{j 2\pi u x}dx$$</p>

퓨리에 변환의 핵심은 어떠한 신호 (<span>$$f(x)$$</span>)는 강도(amplitude, <span>$$F(u)$$</span>)를 가지고 있는 <span>$$e^{j 2\pi u x}$$</span>들의 합으로 이루워 진다. <span>$$e^{j 2\pi u x}$$</span>는 아래와 같은 의미를 가지고 있다.

<p>$$e^{j 2\pi u x} = \text{cos}(2\pi u x) + j \text{sin}(2\pi u x)$$</p>

즉, <span>$$e^{j 2\pi u x}$$</span>는 실수부에 Cos주기 함수와 허수부의 Sin주기 함수로 이루워져 있다.

**정리하사면, Fourier 변환이란 어떠한 신호 (<span>$$f(x)$$</span>)는 주파수(frequency, <span>$$e^{j 2\pi u x}$$</span>)와 강도(amplitude, <span>$$F(u)$$</span>)의 주기신호들의 합으로서 나타낼 수 있다.**

**Fourier 변환의 특징은 모든 주기함수 (<span>$$e^{j 2\pi u x}$$</span>)는 Orthogonal하다는 특징을 가지고 있다.**

<img src="https://imgur.com/Bdo17jG.png" width="400"><br>
그림 참조: <a href="https://ralasun.github.io/deep%20learning/2021/02/15/gcn/">ralasun Blog</a>

그렇다면, Fourier 변환을 Graph에 어떻게 적용할 지 생각해야 한다.  
적용하기 전에 Graph의 특징을 생각해보면, Graph의 Dataset은 real-symmetric Matrix (Node 1-3과 의 관계와 3-1간의 관계는 동일하다 - Local invariance)라는 특징을 가지고 있다.

이러한 real-symmetric matrix에 eigen-value decomposition을 통하여 얻은 eigenvector는 orthonomal하다 라는 특징을 가지고 있다. **즉, Graph-Signal을 eigen-value decomposition을 한다면 강도(amplitude, eigenvalue)와 신호(eigenvector)를 가지는 Fourier Transform형태로 나타낼 수 있는 것을 알 수 있다.**

**Appendix**: <a href="https://wjddyd66.github.io/others/PCA/">real-symmetric matrix eigen-value decomposition</a>

### Graph Laplacian

Graph Convolution을 이해하기 전에 먼저 Graph Laplacian을 이해하여야 한다. Laplacian에 대한 함수는 아래와 같이 정의된다.
<p>$$\triangle f = \bigtriangledown \cdot \bigtriangledown f = \bigtriangledown^2 f$$</p>

즉, Laplace Operator은 Differential Operator로, 벡터 기울기의 발산(Divergence)을 의미합니다.

**Figure 1. Scalar Function**
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2019-08-28_laplacian/noname02.png" width="250"><br>
그림 참조: <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a>

**Figure 2. Gradient**
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2019-08-28_laplacian/noname03.png" width="250"><br>
그림 참조: <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a>

**Figure 3. Divergence**
<img src="https://raw.githubusercontent.com/angeloyeo/angeloyeo.github.io/master/pics/2019-08-28_laplacian/noname04.png" width="250"><br>
그림 참조: <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a>

즉, **Laplacian Operator는 2차 편미분의 값을 의미하며, 실제 함수의 값이 얼마나 큰지 나타내는 값 이다.** 

**이러한 Laplacian Operator를 Graph에 적용하면, "음수: 내기준에서 작은값 -> 주변 큰 값으로 변화하는 정도", "양수: 내기준에서 큰값 -> 주변 작은 값으로 변하는 정도"로 이해하면 생각하기 편하다.** Graph Laplacian Matrix의 Formula는 아래와 같이 표현할 수 있다.

<p>$$\triangle f(v_i) = \sum_{v_j, v_i} [f(v_i) - f(v_j)]$$</p>

위의 수식을 살펴보면 이웃 노드와의 차이를 계산한 것을 확인할 수 있다. 해당 Laplacian Operator는 아래와 같이 단순화 하여 나타낼 수 있다.

![png](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FTanPN%2FbtqHT1bFzmU%2FgB6y6BPgHHDN5qHqct2q71%2Fimg.png)

위에 수식을 살펴보게 되면, Degree matrix로서 나와 이웃한 노드의 차수(Degree)와 주변 노드와의 차이 로서 이루워져 있다. 

**즉, Graph Laplacian Matrix는 다음과 같은 의미를 가지고 있다. (1) Diagnol 원소의 값은 "내 기준 다른 Node에 얼마나 변화를 주는지 나타낸 다." (2) 나머지 원소의 값은 내게 영향을 미치는 다른 Node들을 의미한다.**

또한, 위와 같은 Laplacian Matrix는 아래와 같이 Normalization하여 많이 사용한다.
<p>$$L_{ij} := D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$</p>
<p>$$L_{ij} \begin{cases}
1, & \mbox{if }i=j \mbox{ and } deg(v_i)\neq 0 \\
-\frac{1}{\sqrt{deg(v_i)deg(v_j)}}, & \mbox{if }i\neq j \mbox{ and } v_i \mbox{ is adjacent to} v_j \\
0, & \mbox{otherwise.}\\
\end{cases}$$</p>

### Eigen Decomposition of Graph Laplacian Matrix
    
먼저, Graph Fourier Transform은 Laplacian Matrix를 Eigen-Decomposition으로서 이루워진다. 해당 수식은 아래와 같다.
<p>$$L = U \land U^T$$</p>

해당 수식은 <a href="https://wjddyd66.github.io/others/PCA/">SVD</a>의 수식과 동일하다. 또한, **Laplacian Matrix는 Symmetric Matrix이므로 Eigen-Vector간에 Orthogonal하다는 것을 알 수 있다.**

이러한 Laplacian Matrix의 Eigen Vector(<span>$$U$$</span>)에 대한 의미를 Visualization하면 아래와 같다. 아래 그림은 eigenvector (<span>$$u_2, u_3, u_4, u_8$$</span>)에 대한 eigen value (<span>$$0 < \lambda_2 < \lambda_3 < \lambda_4 < \lambda_8$$</span>)에 graph 노드 (<span>$$f(v_i)$$</span>)를 임베딩 (<span>$$u^T f(v_i)$$</span>)한 결과를 Visualization한 결과이다.

**Figure. Projection on eigenvector of Laplacian Matrix with each graph node**
<img src="https://imgur.com/JWVlsGQ.png" width="400"><br>
그림 참조: <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a>

**위의 Figure를 살펴보면 크게 2가지의 결과를 볼 수 있다. (1) Node간의 거리가 가까울수록 같은 특성(위의 Figure에서 같은 색)을 가질 확률이 높아. (2) Eigen Value(분산)가 크면 클 수록 멀리 있는 Node까지 고려하고, Eigen Value가 작으면 근처 Node간의 관계를 고려한다.**

### Graph Fourier Transform
위에서 정리한 내용을 다시 생각하면 다음과 같다.
- Fourier Transform 
    - (1) Time/spatial domain -> Frequence Domain으로 변환하는 과정이다. 
    - (2) 입력 신호에 대하여 강도(amplitude, eigenvalue)와 신호(eigenvector)를 가지는 신호의 조합으로 나타낼 수 있다.
    - (3) 각 신호는 Cosine or Sine 이므로 Orghogonal 하다는 특징을 가지고 있다.
- Laplacian Operator
    - (1) 미분을 두번 수행하는 연산으로서 발산(Divergence)을 의미한다
    - (2) Graph에서의 Laplacian Matrix는 <span>$$L_{ij} := D^{-\frac{1}{2}} L D^{-\frac{1}{2}}$$</span>로 간단하게 표기되며, 주변 노드와의 변화를 나타내는 값을 가지고 있다.
    - (3) <span>$$L$$</span>을 Eigen-Decomposition하게 되면, <span>$$L = U \land U^T$$</span>로 나타낼 수 있고, <span>$$U$$</span>는 Eigen Vector이면서 Orthogonal하다는 특징을 가지고 있다.
    
Graph Fourier Transform은 아래와 같이 간단히 나타낼 수 있다.

<p>$$ \mathcal{F} (x) = U^T x \text{ - Fourier Transform}$$</p>
<p>$$ \mathcal{F}^{-1} (\hat{x}) = U \hat{x} \text{ - Inverse Fourier Transform}$$</p>
<p>$$\because U^T U = I$$</p>

- <span>$$\hat{x}$$</span>: Result of Fourier Transform
- <span>$$x = \sum_{i} \hat{x_i}u_i$$</span>

위의 수식을 생각해보면 다음과 같다. "<span>$$x$$</span>라는 입력 신호"를 "Orthogonal한 여러 신호의 조합"으로 변환하기 위하여 "Laplacian Matrix (<span>$$L$$</span>)"을 "Eigen-Decomposition (<span>$$U$$</span>)"하여 사용하였다. 
- (1) Laplacian Matrix을 사용한 이유는 "해당 Node뿐만 아니라 주변에 있는 Node까지 고려하기 위해서 이다." 
- (2) <span>$$L = U \land U^T$$</span>에서 <span>$$\land$$</span>의 값을 조절하여 관계를 고려하고자 하는 Node의 범위를 지정할 수 있다.
- (3) <span>$$\mathcal{F} (x) = U^T x$$</span>의 의미를 생각하면 Fourier Transform과 동일하다.
- (4) <span>$$U^T U = I$$</span>

### Spectral Graph Convolution
위에서 정리한 내용을 다시 생각하면 다음과 같다.
- Fourier Transform 
    - (1) Time/spatial domain -> Frequence Domain으로 변환하는 과정이다. 
    - (2) 입력 신호에 대하여 강도(amplitude, eigenvalue)와 신호(eigenvector)를 가지는 신호의 조합으로 나타낼 수 있다.
    - (3) 각 신호는 Cosine or Sine 이므로 Orghogonal 하다는 특징을 가지고 있다.
- Laplacian Operator
    - (1) 미분을 두번 수행하는 연산으로서 발산(Divergence)을 의미한다
    - (2) Graph에서의 Laplacian Matrix는 <span>$$L_{ij} := D^{-\frac{1}{2}} L D^{-\frac{1}{2}}$$</span>로 간단하게 표기되며, 주변 노드와의 변화를 나타내는 값을 가지고 있다.
    - (3) <span>$$L$$</span>을 Eigen-Decomposition하게 되면, <span>$$L = U \land U^T$$</span>로 나타낼 수 있고, <span>$$U$$</span>는 Eigen Vector이면서 Orthogonal하다는 특징을 가지고 있다.
- Graph Fourier Transform
    - (1) <span>$$\mathcal{F} (x) = U^T x \text{ - Fourier Transform}$$</span>
    - (2) <span>$$\mathcal{F}^{-1} (\hat{x}) = U \hat{x} \text{ - Inverse Fourier Transform}$$</span>

**먼저 Filter(h)를 사용하여 어떠한 신호(x)에 의미있는 정보를 찾기 위하여 Fourier Transform을 사용하여 아래 Figure와 같은 process로서 진행하게 된다.**

<img src="https://imgur.com/hnw3IeK.png" width="600"><br>
그림 참조: <a href="https://angeloyeo.github.io/2019/08/28/laplacian.html">angeloyeo Blog</a>

위와 같은 Fourier Transform은 Time/spatial domain -> Frequence Domain으로서 변환하는 과정이므로 해당 과정은 "이전의 값까지의 영향을 고려하여 시스템 출력을 계산" 합니다. 이러한 계산은 Convolution이라 지칭되며, 아래 예시와 수식과 같다.

<p>$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t-\tau) d\tau$$</p>

위의 Fourier Transform을 통한 Convolution결과의 진행 사항은 아래와 같다.
1. 신호와 Filter를 모두 FT를 진행하여 Frequency Domain으로서 변환 한다.
2. FT로 변환된 두 값을 곱한다.
3. 2의 결과를 Inverse FT를 진행한다.

**위의 과정을 간략하게 표현하면 아래와 같다.**
<p>$$g * x = \mathcal{F}^{-1}(\mathcal{F} (g) \odot \mathcal{F} (x))$$</p>
- <span>$$\odot$$</span>: Element-wise 곱셈
- <span>$$g$$</span>: Convolution Filter

**위의 과정을 Graph Fourier Transform을 사용하면 아래와 같다.**
<p>$$g * x = \mathcal{F}^{-1}(\mathcal{F} (g) \odot \mathcal{F} (x))$$</p>
<p>$$= U (U^T x \odot U^T g)$$</p>

**위의 식에서 <span>$$g_{\theta} = diag(U^T g)$$</span> 로서 간략히 치환하면 아래와 같이 정의할 수 있다.**
<p>$$g_{\theta} * x = U g_{\theta} U^T x - (3)$$</p>

### Chebyshev Spectral Convolutional Neural Network (ChebNet)

**<span style="color:red">해당 Section은 이해하지 못하였습니다. 아래 두 링크를 참조하시면 됩니다. (아래 내용은 이해한 부분만 적어두었습니다.)</span>**
- <a href="https://arxiv.org/pdf/1606.09375.pdf">ChebNet Paper</a>
- <a href="https://en.wikipedia.org/wiki/Chebyshev_polynomials">Chebyshev_polynomials</a>

위의 수식 (3)을 생각하게 되면 "non-parametric filter" (<span>$$g_{\theta} = diag(U^T g)$$</span>)을 사용함으로 인하여 크게 2가지 문제가 있다.
1. 공간에 국한되지 않는다. 즉, 퓨리에 방법으로 고유값을 도출하므로, **특정 local 정보를 국한 시킬수 없다. 즉, 이웃 노드 몇개까지만 사용할지 정의할 수 없다.**
2. 데이터의 차원이 커질수록 Computation Resource를 많이 잡아먹게 된다.

해당 문제점을 해결하기 위하여 ChebNet은 K-th order polynomial이 되도록 문제를 해결하였다.

**Chebyshev polynomials는 특정 신호에 대하여 orthogonal polynomials로서 근사화 하는 방법 이다.**  
우리는 위의 식에 Chebyshev polynomial을 사용하면 아래와 같이 근사화 할수 있다고 <a href="https://arxiv.org/pdf/1606.09375.pdf">ChebNet Paper</a>논문에서 증명하였다.

<p>$$g_{\theta^{'}} \approx \sum_{k=0}^K \theta_k^{'} T_k(\tilde{L})x - (4)$$</p>

- <span>$$\tilde{L} = \frac{2}{\lambda_{max}}L - I_N$$</span>
- <span>$$T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x), T_0(x)=1, T_1(x)=x$$</span>

**해당 수식에서 우리가 이해하고 넘어가야 하는 부분은 아래와 같이 2개 이다.**
- Problem: 특정 local 정보를 국한 시킬수 없다. 
    - Solution: K값을 조절하여 local 정보를 국한시킬 수 있다.
    - Why?: 쉽게 "Message passing"을 예시로 들어서 설명하면, Time이 지남에 따라서 주변 정보가 들어오는 것을 Figure로서 보여 주었다. 해당 관점에서 생각하여 Time이 지날수록 (곱셈 연산이 많아질수록) 해당 Node기준으로부터 멀리까지 떨어진 정보까지 포함하게 된다.
    - Example:
        - <span>$$T_2(\tilde{L}) = 2\tilde{L}^2$$</span>
        - <span>$$T_3(\tilde{L}) = 2\tilde{L}^3 - \tilde{L}$$</span>
    
- Problem: Computation Resource를 많이 잡아먹는다. 
    - Solution: <span>$$U$$</span>가 Symmetric Matrix이므로 Computation Resource를 많이 줄일수 있다.
    - Why?: <span>$$L^n = U \land U^T U \land U^T, \ldots = U \land^n U^T, \because UU^T = I$$</span>

### LAYER-WISE LINEAR MODEL

**GCN은 수식 (4)를 변형하여 Layer를 쌓는 형식으로 변환한 것을 제안한다.**

위의 수식(4)에서 <span>$$k=1, \lambda_{max} \approx 2$$</span>을 대입하면 식은 아래와 같이 변한다.

<p>$$g_{\theta}^{'} * x \approx \theta_0^{'} x + \theta_1^{'}T_1(\tilde(L))x$$</p>
<p>$$= \theta_0^{'} x + \theta_1^{'} (L - I_N)x$$</p>
<p>$$= \theta_0^{'} x + \theta_1^{'} (I_N - D^{-\frac{1}{2}} A D^{-\frac{1}{2}} - I_N)x$$</p>
<p>$$= \theta_0^{'} x + \theta_1^{'} - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}x$$</p>

GCN은 위의 수식에서 Parameter의 수를 줄이기 위하여 (<span>$$\theta_0^{'}, \theta_1^{'}$$</span>)을 하나의 (<span>$$\theta$$</span>)로서 표현하였다.

<p>$$g_{\theta}^{'} * x \approx \theta(I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}})x$$</p>

위의 수식의 <span>$$I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$$</span> 의 eigenvalues는 [0, 2] 범위 이다. 논문 저자들은 해당 부분의 range를 [0, 1]로 바꾸어야 stable하게 학습이 가능하였으며, 바꾸는 방법은 아래와 같이 제안하였다.

<p>$$I_N + D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$$</p>
- <span>$$\tilde{A} = A+I_N$$</span>
- <span>$$\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$$</span>


**GCN은 최종적으로 위와 같은 수식을 아래와 같이 General한 Formula로서 소계한다.**
<p>$$Z = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}X \Theta - (3)$$</p>

- <span>$$Z \mathbb{R}^{N \times F}$$</span>: Output
- <span>$$X \mathbb{R}^{N \times C}$$</span>: Input
- <span>$$\Theta \mathbb{R}^{C \times F}$$</span>: Weight


### GRAPH CONVOLUTIONAL NETWORKS

위의 수식 (3)은 Layer-wise linear model에 대한 수식이였다. 해당 수식을 Non-Linear한 Activation을 사용하여 GNN의 Model을 구축한다면 아래와 같은 Formula로서 나타낼 수 있다.

<p>$$H^{(l+1)} = \sigma (\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) - (4)$$</p>

**해당 수식에서 이해하여야 하는 점은 Layer가 쌓일 수록 (l이 커질 수록) 해당 Node로부터 얼만큼 먼 Node까지 고려하는지 지정할 수 있다는 것 이다. 또한, Activation Function을 사용하여 Non-Linearity를 추가하였다. 마지막으로 우리가 기존에 사용하던 DNN처럼 Layer를 쌓으면서서 Back-propagation이 수행 될 수 있다.**

### Appendix. Model Depth

우리는 수식 (4)에서 Layer를 쌓아가면서 얼만큼 먼 Node까지 고려하는지 정할 수 있다. 아래 Figure는 Layer를 쌓아가면서 Model의 Performance를 확인한 결과 이다.

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/RecSys/GCN/1.png)

**해당 결과를 확인하면, 기본적으로 사용하는 수식 (4)처럼 Layer를 쌓아가면, Model의 Performance가 낮아지는 것을 알 수 있다.** 즉, 해당 Node에서 너무많은 주변 정보를 사용하게 되면, 개인의 정보를 잃어버려서 Performance가 낮아지는 것을 알 수 있다.

따라서, 해당 논문의 저자는 CNN의 Residual과 같이 자기 자신의 Node정보를 유지하도록 아래와 같은 Residual Graph Convolution Network를 사용하였다.

<p>$$H^{(l+1)} = \sigma (\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) + H^{(l)} - (5)$$</p>

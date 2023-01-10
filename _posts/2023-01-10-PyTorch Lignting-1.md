---
layout: post
title:  "PyTorch Lightning Ch1-Abstract"
date:   2023-01-10 07:00:20 +0700
categories: [PytorchLightning]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## PyTorch Lightning

**PyTorch Lightning에 대한 정리된 Document가 존재하지 않고, 아직 완성된 것 같지 않습니다. 하지만, 잘 사용하면 충분한 가치가 있다고 생각하여 개인적으로 정리한 Post 입니다.**

### What is PyTorch Lightning
PyTorch Lightning이란 PyTorch 문법을 가지면서 학습 코드를 PyTorch보다 더 효율적으로 작성할 수 있는 파이썬 오픈소스 라이브러리이다.

PyTorch만으로 딥러닝 모델을 만들고 학습할 수 있다. 하지만, CPU, GPU, TPU간의 변경, mixed_precision training(16 bit)등의 복잡한 조건과 반복되는 코드 (training, validation, testing, inference)들을 좀더 효율적으로 추상화 시키자는 목적으로 PyTorch Lightning가 나오게 되었다.

위의 설명은 아래와 같은 그림으로서 표현 가능하다.

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Pytorch_Light/1.png)<br>
그림 출처: <a href="https://visionhong.tistory.com/30">visionhong 블로그</a>

### PyTorch Lightning Example

아래의 예시를 보게 되면, PyTorch Lightning을 살펴보게 되면, 긴 PyTorch Code가 간략하게 정리되는 것을 알 수 있다.

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Pytorch_Light/2.png)<br>
그림 출처: <a href="https://koreapy.tistory.com/1204">koreapy 블로그</a><br>

![png](https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Pytorch_Light/3.png)<br>
그림 출처: <a href="https://koreapy.tistory.com/1204">koreapy 블로그</a><br>

또한, 사용하면서 느끼는 점은, 현재 DeepLearning을 구현하고 학습하는 과정은 반복되는 과정(Training, Validation, Testing, Inference)이 존재하고 어떠한 모델을 사용하여도 같은 과정을 거친다고 알 수 있다.

개인적인 생각으로서 PyTorch Lightning을 사용하여 이러한 과정을 구현하게 되면, 같은 과정의 Code가 같은 곳에 위치하게 되므로, 다른 사람들이 보기 편하므로 Co-Work에서 상당한 강점이 있을 것 같다.

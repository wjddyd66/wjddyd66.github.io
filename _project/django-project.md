---
layout: project_single
title:  "Django project"
slug: "Django Project"
---

### 따봉 Django Project
따봉 Django Project는 실제 존재하는 따릉이 대여소와 기타 요인간의 상관관계분석을 통한 공공자전거 대여소 설치 구역 추천을 하는 시스템입니다.   
팀원: 황정용, 김동혁, 안상민, 장보성, 천지훈, 표종은  
프로젝트 기간: 2주  

### 분석배경
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django1.PNG" height="100%" width="100%" /></div>
2015년 따릉이 사업 시작 이후로 따릉이 가입자 수가 <span style ="color: red">**지속적으로 증가**</span> <br>
특히 이용자 수는 <span style ="color: red">**2016년 이후로 급증**</span>하는 추세를 보임  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django2.PNG" height="100%" width="100%" /></div>
급증하는 이용자 수에 비해 대여소 수가 부족  
2017년 기준 대여소 추가 구축 민원 <span style ="color: red">**173건**</span> <br>

### 목적
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django3.PNG" height="100%" width="100%" /></div>
대여소가 부족한 지역을 파악하는 것이 아닌 이용자 수가 많은 대여소를 파악하고 특징을 추출하여 새로운 대여소 설치시 적절한 위치 제안을 하는 것을 목적으로 한다.  

### 데이터 변수 설정
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django4.PNG" height="100%" width="100%" /></div>
서울 시설공단 공공자전거운영처의 대여소 설치기준을 참고하여 <span style ="color: red">**유동인구, 대학교, 관광명소, 자전거도로**</span>를 변수로 설정  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django5.PNG" height="100%" width="100%" /></div>
공공자전거 이용자수 상위 100개소를 지도에 표시한 결과 공원, 강 주변에 위치  
<span style ="color: red">**공원, 강**</span>의 위치를 변수로 설정  
<a href="https://wjddyd66.github.io/project/2019/07/03/Django-Project-Bike.html">자세한 내용</a>  

### 데이터 전처리
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django6.PNG" height="100%" width="100%" /></div>
<span style ="color: red">**Euclidean**</span>거리 계산 방식 이용  
대여소로부터 최단거리의 “대학교, 자전거도로, 관광명소,  공원, 강”까지의 거리를 도출  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django7.PNG" height="100%" width="100%" /></div>
<span style ="color: red">**Euclidean**</span>거리 계산 방식 이용  
대여소로부터 1.5km내, 관측소들의 유동인구  평균을 도출  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django8.PNG" height="100%" width="100%" /></div>
1. 연속형, 입력변수에<span style ="color: red">MIN-MAX Normalization</span>를 통해 0~1값으로 치환
2. 필요에 따라 연속형 변수 => 범주형 변수로 변경 
3. 결측치 처리: 구간별 중위수로 대체

<a href="https://wjddyd66.github.io/project/2019/07/03/Django-Project-PreData.html">자세한 내용</a>  

### 최종 데이터
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>변수</td><td>설명</td><td>값</td>
	</tr>
	<tr>
		<td><span style ="color: red">Road </span></td>
		<td><span style ="color: red">근접 자전거도로 최단거리 </span></td>
		<td><span style ="color: red">연속형(0~1) </span></td>
	</tr>
	<tr>
		<td>Popular</td>
		<td>근접 명소 최단거리</td>
		<td>연속형(0~1)</td>
	</tr>
	<tr>
		<td>Park</td>
		<td>근접 공원 최단거리</td>
		<td>연속형(0~1)</td>
	</tr>
	<tr>
		<td>River</td>
		<td>근접 강, 하천 최단거리</td>
		<td>연속형(0~1)</td>
	</tr>
	<tr>
		<td>People</td>
		<td>범위 안(1.5 km) 유동인구 평균</td>
		<td>연속형(0~1)</td>
	</tr>
	<tr>
		<td>Univ</td>
		<td>근접 대학교 최단거리</td>
		<td>연속형(0~1)</td>
	</tr>
	<tr>
		<td>Count</td>
		<td>대여소 이용횟수</td>
		<td>범주형(1~3)</td>
	</tr>
</tbody>
</table>
<br>

### Model 선정
<table class="table">
	<tbody>
	<tr>
		<td>모델 명</td><td>정확도(Traing)</td><td>정확도(Test)</td><td>과적합 여부</td>
	</tr>
	<tr>
		<td><span style ="color: red">MLPClassifier</span></td>
		<td><span style ="color: red">61.6 %</span></td>
		<td><span style ="color: red">61 %</span></td>
		<td><span style ="color: red">X</span></td>
	</tr>
	<tr>
		<td>GradientBoostingClassifie</td>
		<td>54.6 %</td>
		<td>52.2 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>K-NN</td>
		<td>57 %</td>
		<td>50 %</td>
		<td>O</td>
	</tr>
	<tr>
		<td>Decision Tree</td>
		<td>54 %</td>
		<td>50 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>SVM</td>
		<td>53 %</td>
		<td>52 %</td>
		<td>X</td>
	</tr>
	<tr>
		<td>Random Forest</td>
		<td>99.2 %</td>
		<td>52 %</td>
		<td>O</td>
	</tr>
</tbody>
</table>
<br>

정확도가 가장높고, 과적합 하지 않은 모델 선정<span style ="color: red">**MLPClassifier**</span>가 모델로 선정  
MLPClassifier  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django9.PNG" height="100%" width="100%" /></div>
1. 모델 구조 및 가정에서 최소의 요구를 가지고 있는 광범위한 예측 모델과 근사
2. 모델  해석 가능성이 낮지만 좋은 <span style ="color: red">**예측력**</span>을 확보할 수 있음  

<a href="https://wjddyd66.github.io/project/2019/07/03/Django-Project-Model.html">자세한 내용</a>  

### Model Parameter 선택
<table class="table">
	<tbody>
	<tr>
		<td>Parameter</td><td>설명</td><td>값</td>
	</tr>
	<tr>
		<td><span style ="color: red">Hidden_Layer_Sizer</span></td>
		<td><span style ="color: red">Hidden Layer 크기 설정</span></td>
		<td><span style ="color: red">(10,10,30)</span></td>
	</tr>
	<tr>
		<td>max_iter</td>
		<td>최대 반복 횟수</td>
		<td>3000</td>
	</tr>
	<tr>
		<td>alpha</td>
		<td>L2 Regulation penalty</td>
		<td>0.0001</td>
	</tr>
	<tr>
		<td>activation</td>
		<td>활성 함수</td>
		<td>relu</td>
	</tr>
	<tr>
		<td>solver</td>
		<td>weight optimizer</td>
		<td>adam</td>
	</tr>
	<tr>
		<td>learning_rate</td>
		<td>Schedule for weight updates</td>
		<td>adaptive</td>
	</tr>
</tbody>
</table>
<br>
MLP Classifer 모델의 Parameter를 반복적으로 변경  
<span style ="color: red">**최적의 변수 값을 선정 (정확도: 61~72%)**</span><br>

<a href="https://wjddyd66.github.io/project/2019/07/03/Django-Project-Parameter.html">자세한 내용</a>  

### 분석결과
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django10.PNG" height="100%" width="100%" /></div><br><br>
시연영상  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/6cb454d359ad411ca786bb5d035f414f" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<a href="https://wjddyd66.github.io/project/2019/07/03/Django-Project-Result.html">자세한 내용</a>  

### 결론
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django11.PNG" height="100%" width="100%" /></div><br>
1. 적합도 상위조건
 - 강과 가까울수록 적합도가 높은 경향을 보인다.
 - 자전거 도로와 인접할 수록 적합도가 높다.

2. 등급분류기준
 - 종합적  판정  결과  <span style ="color: red">상위  20% = “상” </span>, <span style ="color: blue">하위  10%  = “하”</span>
 - ‘상’   등급  :  지정  위치에  대여소  설치  추천
 - ‘중’   등급:  내부   검토에  의해   설치
 - ‘하’   등급:   설치  지양

### 개선 방향
1. 데이터 확장
 - 타  지역으로 확장 시 해당시의 도로, 공원,  유동인구 등의 데이터가 필요
 - 운용  가능한  데이터의  다양성  확보  필요
2. 사업분야의 확장
 - 공공 전동 킥보드 등의  대여 서비스로 확장 & 적용을  기대
 - 도로  및  시설물의 인프라  확장에  정보 를  제공
3. 기상, 비정량적 데이터 고려
 - 시시각각 변하는 기상 데이터와의  연동 시스템 구축
 - SNS등을  활용한  대중의  개별적  감정상태  반영

### 참고 사이트
<a href="https://data.seoul.go.kr">서울시 열린데이터광장</a>  
<a href="https://cloud.google.com/maps-platform/?hl=ko">구글맵</a>  
<a href="http://www.sisul.or.kr/open_content/main">서울시설공단</a>  
<a href="https://www.si.re.kr">서울 연구원</a>  
<a href="https://www.bikeseoul.com">서울자전거 따릉이 –무인대여시스템</a> 외 다수  

<hr>
참조: <a href="https://github.com/wjddyd66/Project/tree/master/Django">원본 Project Folder</a>
궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


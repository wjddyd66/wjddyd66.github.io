---
title: "About"
layout: single
permalink: /about/
author_profile: true
---

### AI Developer

2019년에 졸업을 하여 현재 AI분야에 대한 전문가가 되기위하여 노력하고 있는 Programmer입니다.  
AI에서도 Vision분야, DeepLearning 분야에 대해 관심이 많고 또한 Workflow 구현을 위한 Infra에 대해서도 관심이 많습니다.  

#### Career

* [한국외국어 대학교](http://www.hufs.ac.kr//) :: 2013. 03. ~ 2020.20.(졸업 예정)
* [클라우드 기반 빅데이터분석 및 자바 딥러닝 개발자(국비교육)](http://kiccampus.co.kr//) :: 2019. 01. ~ 2019. 07.
* [2019 머신러닝 스터디 잼 심화반](https://sites.google.com/view/studyjamkr/) :: 2019. 06. ~ 2019. 07.
* [PopcornSAR](https://popcornsar.com/main/home) :: 2019. 08. ~ Today.

#### certificate
- 정보처리기사 [2018/05/25]: 18201230170A
- 정보통신기사 [2018/08/31]: 18-71-0155
- ADSP [2019/05/25]
- TOEIC [ ~ 2021/02/10]: 690


#### Skill

* Language - JAVA / Python / R
* Database -  MySQL / NoSQL(Mongo DB)
* Infra - Docker / Kubernetes / Kubeflow
* AI - Tensorflow(1.x) / Pytorch
* Vision - OpenCV (Python)
* Windows & Linux Platform
* Web Based Server & Tools - JS / JSP / JQuery / Spring / Django

<br>
---

### PROJECT

---

#### 따봉 Django Project
따봉 Django Project는 실제 존재하는 따릉이 대여소와 기타 요인간의 상관관계분석을 통한 공공자전거 대여소 설치 구역 추천을 하는 시스템입니다.   
팀원: 황정용, 김동혁, 안상민, 장보성, 천지훈, 표종은  
프로젝트 기간: 2주  

#### 분석배경
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django1.PNG" height="100%" width="100%" /></div>
2015년 따릉이 사업 시작 이후로 따릉이 가입자 수가 <span style ="color: red">**지속적으로 증가**</span> <br>
특히 이용자 수는 <span style ="color: red">**2016년 이후로 급증**</span>하는 추세를 보임  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django2.PNG" height="100%" width="100%" /></div>
급증하는 이용자 수에 비해 대여소 수가 부족  
2017년 기준 대여소 추가 구축 민원 <span style ="color: red">**173건**</span> <br>

#### 목적
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django3.PNG" height="100%" width="100%" /></div>
대여소가 부족한 지역을 파악하는 것이 아닌 이용자 수가 많은 대여소를 파악하고 특징을 추출하여 새로운 대여소 설치시 적절한 위치 제안을 하는 것을 목적으로 한다.  

#### 데이터 변수 설정
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django4.PNG" height="100%" width="100%" /></div>
서울 시설공단 공공자전거운영처의 대여소 설치기준을 참고하여 <span style ="color: red">**유동인구, 대학교, 관광명소, 자전거도로**</span>를 변수로 설정  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django5.PNG" height="100%" width="100%" /></div>
공공자전거 이용자수 상위 100개소를 지도에 표시한 결과 공원, 강 주변에 위치  
<span style ="color: red">**공원, 강**</span>의 위치를 변수로 설정  
<a href="https://wjddyd66.github.io/project/Django-Project-Bike/">자세한 내용</a>  

#### 데이터 전처리
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

<a href="https://wjddyd66.github.io/project/Django-Project-PreData/">자세한 내용</a>  

#### 최종 데이터
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

#### Model 선정
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

<a href="https://wjddyd66.github.io/project/Django-Project-Model/">자세한 내용</a>  

#### Model Parameter 선택
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

<a href="https://wjddyd66.github.io/project/Django-Project-Parameter">자세한 내용</a>  

#### 분석결과
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django10.PNG" height="100%" width="100%" /></div><br><br>
시연영상  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/6cb454d359ad411ca786bb5d035f414f" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<a href="https://wjddyd66.github.io/project/Django-Project-Result/">자세한 내용</a>  

#### 결론
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/Django11.PNG" height="100%" width="100%" /></div><br>
1. 적합도 상위조건
 - 강과 가까울수록 적합도가 높은 경향을 보인다.
 - 자전거 도로와 인접할 수록 적합도가 높다.

2. 등급분류기준
 - 종합적  판정  결과  <span style ="color: red">상위  20% = “상” </span>, <span style ="color: blue">하위  10%  = “하”</span>
 - ‘상’   등급  :  지정  위치에  대여소  설치  추천
 - ‘중’   등급:  내부   검토에  의해   설치
 - ‘하’   등급:   설치  지양

#### 개선 방향
1. 데이터 확장
 - 타  지역으로 확장 시 해당시의 도로, 공원,  유동인구 등의 데이터가 필요
 - 운용  가능한  데이터의  다양성  확보  필요
2. 사업분야의 확장
 - 공공 전동 킥보드 등의  대여 서비스로 확장 & 적용을  기대
 - 도로  및  시설물의 인프라  확장에  정보 를  제공
3. 기상, 비정량적 데이터 고려
 - 시시각각 변하는 기상 데이터와의  연동 시스템 구축
 - SNS등을  활용한  대중의  개별적  감정상태  반영

#### 참고 사이트
<a href="https://data.seoul.go.kr">서울시 열린데이터광장</a>  
<a href="https://cloud.google.com/maps-platform/?hl=ko">구글맵</a>  
<a href="http://www.sisul.or.kr/open_content/main">서울시설공단</a>  
<a href="https://www.si.re.kr">서울 연구원</a>  
<a href="https://www.bikeseoul.com">서울자전거 따릉이 –무인대여시스템</a> 외 다수  

참조: <a href="https://github.com/wjddyd66/Project/tree/master/Django">원본 Project Folder</a>

---

#### BOM AIR(Best Of Most Airline & Rent Car) Spring Project
BOM AIR Spring Project는 실제 항공사들이 서비스하는 Flight Booking + Car Rent를 목표로 하여 만든 프로젝트 입니다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo2.PNG" height="100%" width="100%" /></div>  
팀원: 황정용, 김동혁, 안상민, 장보성, 천지훈, 표종은  
프로젝트 기간: 2주  

#### Use Case Diagram
실제 이용자가 사용하는 경우의 순서도를 Use Case Diagram으로 나타내었다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo3.PNG" height="100%" width="100%" /></div>  
<br>
사용자의 경우 크게 3가지로 나누었다.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>사용자</td><td>사용 가능 기능</td>
	</tr>
	<tr>
		<td>Login을 하지 않은 사용자</td><td>
		<ul>
			<li>회원가입</li>
			<li>아이디, 비밀번호 찾기</li>
			<li>비행기 조회</li>
			<li>렌트카 조회</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>Login이 되어있는 사용자</td><td>
		<ul>
			<li>회원정보 수정, 탈퇴</li>
			<li>비행기 티켓 예약</li>
			<li>(티켓 예약 후)웹 체크인</li>
			<li>예약 정보 상세보기</li>
			<li>렌트카 예약</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>관리자</td><td>
		<ul>
			<li>매출현황 조회</li>
			<li>비행기 등록</li>
			<li>렌트카 등록, 수정, 삭제</li>
			<li>공지사항 등록, 수정, 삭제</li>
		</ul>
		</td>
	</tr>
	</tbody>
</table>
<br>

각각의 사용자는 Session을 이용하여 판별하였다.  
<br>

#### EXERD Diagram
DB는 Maria DB를 사용하였고, 프로젝트를 위한 DB의 설계는 아래의 그림과 같이 설계하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo4.PNG" height="100%" width="100%" /></div>  
<br>
#### Project Main Page
Project의 Main Page이다.  
각각의 사용자를 Session에 이용하여 사용자를 판별하고 특정 사용자(관리자)의 경우 Header에 옵션을 주어 특정 기능이 보이도록 설계하였다.  
Main Page에서 바로 최근 공지사항을 보이도록 하였고 공지사향의 내용이 길 경우 앞의 내용만 보이게 하여 화면이 깨지지 않게 구성하였다.  
Project의 Main 기능인 항공권 예매와 렌트카 예매의 경우 바로 Main 화면에서 구매 가능하도록 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo5.PNG" height="100%" width="100%" /></div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo6.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-MainPage">자세한 내용</a>  
<br>
#### 회원가입
회원 가입 같은 경우 회원의 ID를 Primary Key로서 사용하기 때문에 중복체크를 하여 확인하였다.  
주소 입력같은 경우 모든 주소를 DB에 넣는 작업이 커질것을 우려하여 다음 API를 사용하여 주소를 입력하게 하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo7.PNG" height="100%" width="100%" /></div>  

#### ID, 비밀번호 찾기
계정의 정보를 잃어버려 ID, 비밀번호 찾기를 해야 하는경우, ID는 바로 보여줘도 상관없지만, 비밀번호 같은 경우 중요한 정보이므로 가입되어있는 Mail로 발송하는 식으로 구현하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo8.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-ID">자세한 내용</a>  
<br>

#### 공지사항
공지사항인 경우 Main Page에서 보여지는 부분은 Ajax로서 처리하였다.  
공지사항은 등록 수정, 삭제의 경우는 Session의 값이 관리자일때만 가능하게 하였다.  
사용자의 경우 공지사항을 볼 수 밖에 없게 구성하였다.  
공지사항의 보여주는 부분을 일정하게 보여주기 위하여 Pagination을 적용하였다.  
많은 공지사항일 경우 원하는 정보를 보기 위하여 검색기능을 넣었다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo9.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-Ge">자세한 내용</a>  
<br>

#### 렌트카 등록
관리자의 경우 렌트카의 정보를 입력한 뒤 DB에 저장하게 되었다.  
렌트카일 경우 차량의 사진을 File Upload하여 실제 저장소에 올라가게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo10.PNG" height="100%" width="100%" /></div>  

#### 렌트카 예약
사용자의 경우 관리자가 등록한 렌트카를 예약할 수 있게 구성하였다.  
예약을 하고 예약정보를 확인할 수 있게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo11.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-Rent">자세한 내용</a>  
<br>

#### 관리자- 항공편 등록, 매출 확인
관리자의 경우 게시판에서 특정한 작업원 권한을 부여받을 뿐 아니라, 항공편 생성 및 매출을 확인 가능하다.  
매출의 경우 일단위는 달력 UI를 활용하여 한번에 볼 수 있게 구성하였고 매출에 대한 자세한 내용은 직접 들어가서 확인 가능하게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo12.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-Admin">자세한 내용</a>  
<br>

#### 비행기 예약
관리자가 비행기를 등록하고 나면, 등록되어있는 비행기를 사용자는 예약을 할 수 있다.  
비행기의 조회는 Login이 되어있지 않은 사용자도 가능하지만, 예약은 불가능하게 구성하였다.  
비행기를 예약하고 나면, 조회를 하여 예약정보를 확인할 수 있고 체크인 기능이 활성화 되도록 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo13.PNG" height="100%" width="100%" /></div>  

#### 체크인
체크인이란 예약 몇일전에 사용자가 직접 좌석을 고를 수 있는 System이다.  
Class(일반, 비지니스, VIP)안에서 좌석을 고를 수 있으면 체크인을 하게 되면 티켓을 확인하여 Print하거나 E-mail로 발송하여 티켓을 확인할 수 있게 구성하였다.  
<br>
체크인 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo14.PNG" height="100%" width="100%" /></div>  
<br>
티켓 확인 환면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo15.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-Check">자세한 내용</a>  
<br>

#### 스케줄러
비행기의 티켓의 정보는 비행기가 도착하고 나서는 쓸모없는 Data가 된다.  
매일매일 생기는 이러한 쓸모없는 많은 정보를 관리자가 삭제하는 것이 아닌 일정 시간기준으로 자동으로 작업을 시키기 위하여 스케줄러를 사용하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo16.PNG" height="100%" width="100%" /></div>  
<a href="https://wjddyd66.github.io/project/Spring-Project-Sche">자세한 내용</a>  

<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본 Project Folder</a><br>

---

#### PROJECT '드래곤헌터 FOR KAKAO'
* Main Promotion
  
  <iframe width="700" height="480" src="https://www.youtube.com/embed/82HsgOqylOc" frameborder="0" allowfullscreen></iframe>
* CBT Promotion
  
  <iframe width="700" height="480" src="https://www.youtube.com/embed/mbzc_Lamt7I" frameborder="0" allowfullscreen></iframe>
* '드래곤헌터' 서버 개발 : 프로젝트 초기부터 작업하여 런칭, 서비스 운영까지 참여
* C++/MSSQL로 작업 (Windows 플랫폼)
* 스테이지, 강화/초월, 미션처리, PVP, 무한의숲 등 비즈니스 로직 개발
* 사내 공용플랫폼 시스템과의 연동작업(아이템전송, 이벤트연동, 로깅)
* 운영툴 제작 (ASP.NET, MySQL)
* 유저 데이터 조회, 랭킹 조회, 아이템, 스테이지 조회 등 운영툴의 모든 기본기능을 구현
* 아이템 발송 기능, 유저 데이터 수정 등 데이터 입력/수정 기능을 구현
* 실제 서비스 이후 유지보수 및 추가개발 작업 담당
* SVN 서버 관리, 테스트서버 설치/유지관리와 서버설정 등의 개발인프라 관리 담당

---
<br>

### Paper

---

#### 자연어처리 기술을 활용한 생성형 문서요약
실제로 논문을 정식 학회에 등록한 것이 아닌 학교 졸업논문을 작성하였습니다.  
<a href="http://ice.hufs.ac.kr/">한국외국어대학교 정보통신학과 홈페이지</a>에 접속하면 확인할 수 있으나 Login해야 하는 번거로움이 있어 간단히 볼 수 있게 Post하였습니다.  

<embed src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/projects/paper.pdf" width="800px" height="1000px"/>

---
layout: project_single
title:  "Spring project"
slug: "Spring Project"
---
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo.png" height="100%" width="100%" /></div>



### BOM AIR(Best Of Most Airline & Rent Car) Spring Project
BOM AIR Spring Project는 실제 항공사들이 서비스하는 Flight Booking + Car Rent를 목표로 하여 만든 프로젝트 입니다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo2.PNG" height="100%" width="100%" /></div>  
팀원: 황정용, 김동혁, 안상민, 장보성, 천지훈, 표종은  
프로젝트 기간: 2주  

### Use Case Diagram
실제 이용자가 사용하는 경우의 순서도를 Use Case Diagram으로 나누었다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo3.PNG" height="100%" width="100%" /></div>  
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

### EXERD Diagram
DB는 Maria DB를 사용하였고, 프로젝트를 위한 DB의 설계는 아래의 그림과 같이 설계하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo4.PNG" height="100%" width="100%" /></div>  
<br>
### Project Main Page
Project의 Main Page이다.  
각각의 사용자를 Session에 이용하여 사용자를 판별하고 특정 사용자(관리자)의 경우 Header에 옵션을 주어 특정 기능이 보이도록 설계하였다.  
Main Page에서 바로 최근 공지사항을 보이도록 하였고 공지사향의 내용이 길 경우 앞의 내용만 보이게 하여 화면이 깨지지 않게 구성하였다.  
Project의 Main 기능인 항공권 예매와 렌트카 예매의 경우 바로 Main 화면에서 구매 가능하도록 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo5.PNG" height="100%" width="100%" /></div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo6.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/26/Spring-Project-MainPage.html>  
<br>
### 회원가입
회원 가입 같은 경우 회원의 ID를 Primary Key로서 사용하기 때문에 중복체크를 하여 확인하였다.  
주소 입력같은 경우 모든 주소를 DB에 넣는 작업이 커질것을 우려하여 다음 API를 사용하여 주소를 입력하게 하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo7.PNG" height="100%" width="100%" /></div>  

### ID, 비밀번호 찾기
계정의 정보를 잃어버려 ID, 비밀번호 찾기를 해야 하는경우, ID는 바로 보여줘도 상관없지만, 비밀번호 같은 경우 중요한 정보이므로 가입되어있는 Mail로 발송하는 식으로 구현하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo8.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-ID.html>  
<br>
### 공지사항
공지사항인 경우 Main Page에서 보여지는 부분은 Ajax로서 처리하였다.  
공지사항은 등록 수정, 삭제의 경우는 Session의 값이 관리자일때만 가능하게 하였다.  
사용자의 경우 공지사항을 볼 수 밖에 없게 구성하였다.  
공지사항의 보여주는 부분을 일정하게 보여주기 위하여 Pagination을 적용하였다.  
많은 공지사항일 경우 원하는 정보를 보기 위하여 검색기능을 넣었다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo9.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-Ge.html>  
<br>
### 렌트카 등록
관리자의 경우 렌트카의 정보를 입력한 뒤 DB에 저장하게 되었다.  
렌트카일 경우 차량의 사진을 File Upload하여 실제 저장소에 올라가게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo10.PNG" height="100%" width="100%" /></div>  

### 렌트카 예약
사용자의 경우 관리자가 등록한 렌트카를 예약할 수 있게 구성하였다.  
예약을 하고 예약정보를 확인할 수 있게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo11.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-Rent.html>  
<br>
### 관리자- 항공편 등록, 매출 확인
관리자의 경우 게시판에서 특정한 작업원 권한을 부여받을 뿐 아니라, 항공편 생성 및 매출을 확인 가능하다.  
매출의 경우 일단위는 달력 UI를 활용하여 한번에 볼 수 있게 구성하였고 매출에 대한 자세한 내용은 직접 들어가서 확인 가능하게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo12.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-Admin.html>  
<br>
### 비행기 예약
관리자가 비행기를 등록하고 나면, 등록되어있는 비행기를 사용자는 예약을 할 수 있다.  
비행기의 조회는 Login이 되어있지 않은 사용자도 가능하지만, 예약은 불가능하게 구성하였다.  
비행기를 예약하고 나면, 조회를 하여 예약정보를 확인할 수 있고 체크인 기능이 활성화 되도록 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo13.PNG" height="100%" width="100%" /></div>  

### 체크인
체크인이란 예약 몇일전에 사용자가 직접 좌석을 고를 수 있는 System이다.  
Class(일반, 비지니스, VIP)안에서 좌석을 고를 수 있으면 체크인을 하게 되면 티켓을 확인하여 Print하거나 E-mail로 발송하여 티켓을 확인할 수 있게 구성하였다.  
<br>
체크인 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo14.PNG" height="100%" width="100%" /></div>  
<br>
티켓 확인 환면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo15.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-Check.html>  
<br>
### 스케줄러
비행기의 티켓의 정보는 비행기가 도착하고 나서는 쓸모없는 Data가 된다.  
매일매일 생기는 이러한 쓸모없는 많은 정보를 관리자가 삭제하는 것이 아닌 일정 시간기준으로 자동으로 작업을 시키기 위하여 스케줄러를 사용하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo16.PNG" height="100%" width="100%" /></div>  
자세한 내용:<https://wjddyd66.github.io/spring/2019/06/27/Spring-Project-Sche.html>  

<hr>
참조:<https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
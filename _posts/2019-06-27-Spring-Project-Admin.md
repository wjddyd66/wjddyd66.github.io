---
layout: post
title:  "Spring-Project-관리자"
date:   2019-06-27 08:30:00 +0700
categories: [Project]
---

###  관리자 권한
관리자의 권한으로서 다른 기능에서 많이 적용시켰지만, 가장 큰 Main 기능 2가지는 비행기 생성과 매출확인이다.  
관리자의 비행기 생성을 위한 DB의 구조는 아래와 같다.  
노선을 위한 DB  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Admin1.PNG" height="100%" width="100%" /></div><br>
항공 생성을위한 DB  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Admin2.PNG" height="100%" width="100%" /></div><br>
###  관리자 권한 - 비행기 생성
항공기 생성시 고려해야 할 점은 고객이 예매를 했을경우 Ticket이 각각 다 달라야 한다.  
이를 위하여 항공기의 이름, 출발시간, 도착지, 좌석번호등 여러가지 정보를 조합하여 유일한 Ticket 번호를 생성하게 되었다.  
매출 확인 및 비행기의 정보가 필요한 경우 생성한 Ticket번호로서 구별하게 되었다.  
노선생성시 각 비행기의 출발 년도, 월을 선택하고 노선정보를 입력하게 되면 임의로 1달치 비행기 노선 정보가 생성한다고 가정하고 실시하였다.  
비행기 노선생성을 위한 코드  
```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<link rel="stylesheet" href="resources/css/mybook.css">
</head>
<body class="basicFont container">
<div class="logo"><a href="index.jsp"><img src="resources/images/bomair_logo.png"/></a></div>
<br><br><br>
* 도시 항공노선 정보 입력 * <p/>
<script type="text/javascript" src="http://code.jquery.com/jquery-3.2.0.min.js" ></script>
<script type="text/javascript">
$(document).ready(function(){
  $('#btn_go').click(function(){
		var aa = $('#month').val();
		var bb = $('#year').val();
		$('input[name=o_sdate]').attr('value', bb+'-'+aa+'-01');
		
		if($('#l_code').val() === ''){
			alert("도시를 입력해주세요");
		}else if(bb === ''){
			alert("년도를 입력해주세요");				
		}else if(aa === ''){
			alert("월을 입력해주세요");				
		}else{
			
		if($('#l_code').val() === 'CJU'){
		$('input[name=o_soyo]').attr('value','70');
		//var bb = $('#o_sdate').val();
		//alert(bb);
		}else if($('#l_code').val() === 'NPT'){
			$('input[name=o_soyo]').attr('value','135');			
		}else if($('#l_code').val() === 'KIX'){
			$('input[name=o_soyo]').attr('value','95');			
		}else if($('#l_code').val() === 'FUK'){
			$('input[name=o_soyo]').attr('value','80');			
		}else if($('#l_code').val() === 'HKG'){
			$('input[name=o_soyo]').attr('value','220');			
		}else if($('#l_code').val() === 'BKK'){
			$('input[name=o_soyo]').attr('value','350');			
		}else if($('#l_code').val() === 'BKI'){
			$('input[name=o_soyo]').attr('value','410');			
		}else if($('#l_code').val() === 'WO'){
			$('input[name=o_soyo]').attr('value','270');			
		}else if($('#l_code').val() === 'JFK'){
			$('input[name=o_soyo]').attr('value','995');			
		}	
		alert($('#l_code').val() + "공항 " +  bb + "년 " + aa + "월의 db추가가 완료 되었습니다");

  		$("form:first").submit();
		}
  });
  
});
</script>

<form action="insert" method="post">
<!-- 
노선코드 : <input type="text" name="l_code" id="l_code"><br>
 -->

도시선택 : <select name="l_code" id=l_code>
    <option value="">도시선택</option>
    <option value="CJU">제주도</option>
    <option value="NPT">도쿄</option>
    <option value="KIX">오사카</option>
    <option value="FUK">후쿠오카</option>
    <option value="HKG">홍콩</option>
    <option value="BKK">방콕</option>
    <option value="BKI">코타키나발루</option>
    <option value="WO">블라디보스토크</option>
    <option value="JFK">뉴욕</option>
</select><br>

년 : <input type="text" name="year" id="year" value="2019"><br>
월 : <input type="text" name="month" id="month"><br>
<input type="hidden" name="air_name">
<input type="hidden" name="o_sdate" id="o_sdate">
<input type="hidden" name="o_price"><br>
<input type="hidden" name="o_soyo" id="o_soyo">
<input type="hidden" name="o_stime">


<br>
<input type="button" value="추가" id="btn_go" >
<a href="list">리스트보기</a>
</form>

<br><br><br>
<hr class="hrCss">
<br>
------------------------ 노선 정보 ---------------------<br>
제주(CJU)<br>
도쿄(NPT)<br>
오사카(KIX)<br>
후쿠오카(FUK)<br>
홍콩(HKG)<br>
방콕(BKK)<br>
코타키나발루(BKI)<br>
블라디보스토크(WO)<br>
뉴욕(JFK)

</body>
</html>
```
<br>
추가한 노선은 확인할 수 있게 구성을 하였다.  
```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<link rel="stylesheet" href="resources/css/mybook.css">
</head>
<body class="basicFont container">
<div class="logo"><a href="index.jsp"><img src="resources/images/bomair_logo.png"/></a></div>
<br>
<Br>

** 노선정보 목록 ** <br>
<a href="insert">노선 추가</a><br>
<table border="1">
	<tr>
		<th>노선코드</th><th>항공편명</th><th>출발날짜</th><th>가격</th><th>소요시간</th><th>출발시각</th>
	</tr>

	<c:forEach var="m" items="${list }">
		<tr>
			<td>${m.l_code }</td>
			<td>${m.air_name }</td>
			<td>${m.o_sdate }</td>
			<td>${m.o_price }</td>
			<td>${m.o_soyo }</td>
			<td>${m.o_stime }</td>			
		</tr>
	</c:forEach>
</table>
</body>
</html>
```
비행기 노선을 생성 Page  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Admin3.PNG" height="100%" width="100%" /></div><br>
비행기 노선 생성 확인 Page  

<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Admin4.PNG" height="100%" width="100%" /></div><br>

###  관리자 권한 - 매출확인
매출확인의 경우 관리자가 보기 편하게 달력 UI를 통하여 하루마다 얼마의 매출이있는지를 표시하였다.  
표시되는 가격은 하루 총 매출이고 달력 클릭시 상세한 매출 내역과 기간을 정하여 매출을 확인할 수 있게 구성하였다.  
매출가격 확인은 Ticket안에 있는 정보를 활용하였다.  
달력 UI는 JavaScript를 활용하여 적용하였다.  
```jsp
<%@page import="java.util.ArrayList"%>
<%@page import="java.util.List"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8' />

<link href='resources/css/fullcalendar.min.css' rel='stylesheet' />
<link href='resources/css/fullcalendar.print.min.css' rel='stylesheet'
	media='print' />
<script src='resources/js/moment.min.js'></script>
<script src='resources/js/jquery.min.js'></script>
<script src='resources/js/fullcalendar.min1.js'></script>
<script src='resources/js/ko.js'></script>
<div style="text-align: center"><a href="index.jsp"><img src="resources/images/bomair_logo.png"/></a></div>
<br><br><br>
<script> 

//if(session.getAttribute("idkey") !== null){
$(document).ready(function() {
	$('#calendar').fullCalendar({
		editable: true,
		eventLimit: true,
		locale:"ko",
	header: { left: 'prev,next today', center: 'title', right: 'month,basicWeek,basicDay' }, 
	defaultDate: new Date(), 
	navLinks: true,  
	events: [  
		
		<%=(String) request.getAttribute("list")%>

		
		] 
		
	}); 
	}); 
//}else{
	//alert("로그인 후에 이용 하실수있습니다")
//}
	
	</script>
<style>  
body {
	margin: 40px 10px;
	padding: 0;
	font-family: "Lucida Grande", Helvetica, Arial, Verdana, sans-serif;
	font-size: 14px;
}

#calendar {
	max-width: 900px;
	margin: 0 auto;
}
</style>
</head>
<body>
	<div id='calendar'></div>
</body>
</html>



```
관리자 매출확인 - 달력 UI로 확인하기, 하루하루 매출 확인, 전체매출 확인하기  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/bcee0dea0bd0443d87af55977c0fabb6" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
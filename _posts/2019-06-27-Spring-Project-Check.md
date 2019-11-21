---
layout: post
title:  "Spring-Project-비행기 예약 & 체크인"
date:   2019-06-27 09:00:00 +0700
categories: [Project]
---

###  비행기 예약-Setting
비행기의 예약인 경우 Main Page에서 출발지, 도착지, 일정, 인원수를 입력하게 되면 그 정보에 맞는 비행기 편을 보여주게 된다.  
예약을 하게 되면 예약 페이지를 보여주게 되고 정보를 확인할 수 있게 구성하였다.  
티켓의 경우 체크인을 하여야지 Mail 발송 혹은 Print할 수 있는 Ticket을 보여주게 된다.  
비행기 예약을 위한 DB구성은 아래와 같다.  


<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Book1.PNG" height="100%" width="100%" /></div><br>
<br>

###  비행기 예약
비행기의 경우 왕복과 편도가 존재하게 된다.  
왕복과 편도는 같은 기능을 하나 서로 다른정보를 많이 가지고 있으므로 분리하여서 작성하게 되었다.  
편도 관한 Code  
```jsp
<!-- 티켓 예약 -->
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<%@ taglib prefix="fn" uri="http://java.sun.com/jsp/jstl/functions" %>
<%
	System.out.print(request.getParameter("num"));
%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<script type="text/javascript" src="http://code.jquery.com/jquery-3.2.0.min.js" ></script>
<link href="https://fonts.googleapis.com/css?family=Noto+Sans+KR:400,900'" rel="stylesheet">
<script type="text/javascript">
$(document).ready(function(){
	$('#btnComp').click(function(){
		  var result = confirm('정말 예약하시겠습니까?'); 
		  if(result) { 
			  //$('input[name=count]').attr('value', count);
			$("form:first").submit();
			 //alert($('input[name=maxT]').val());
		  } else { 
			  location.replace("airinfo"); 			  
		  }

	  });
});
function gogo(start, end, time, a_name, grade){
	
	$('input[name=start]').attr('value', start);
	$('input[name=end]').attr('value', end);
	$('input[name=time]').attr('value', time);
	$('input[name=a_name]').attr('value', a_name);
	$('input[name=grade]').attr('value', grade);
	
	//alert($('input[name=a_name]').val());
	
	aa();
	
}

function aa(){
	
	var air_price = $(":input:radio[name=radioTxt]:checked").val() * ${airbean.people };
	//alert(numberWithCommas(air_price));

	var air_price1 = numberWithCommas(air_price) + " KRW"
	//alert(air_price);
	$('#ap').text(air_price1);
	
	var gong = ${airbean.people * 28600} ;
	var gong1 = numberWithCommas(gong) + " KRW";
	$('#gong').text(gong1);
	
	var you = ${airbean.people * 11800} 
	var you1 = numberWithCommas(you) + " KRW";
	$('#you').text(you1);
	
	var tot = air_price + gong + you;
	var tot1 = numberWithCommas(tot) + " KRW";
	$('#tot').text(tot1);
	
	$('input[name=pay]').attr('value', tot);
}


function numberWithCommas(x) {
    return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}


</script>
</head>
<body style = "text-align: center; font-family: 'Noto Sans KR', sans-serif;">
<br>
<br>
<div align="center">
<img class="brand-logo-light" src="resources/images/bomair_logo.png" style="width:510px; height:100px">
<table  rules="none" style="background-color: #7cc67c">
	<tr>
		<td> &nbsp;&nbsp;가는편&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| </td>
		<td colspan="3">&nbsp;&nbsp;&nbsp;&nbsp;인천 (ICN) ▶  
		<c:choose>
		<c:when test="${airbean.l_code eq 'CJU' }">제주</c:when>
		<c:when test="${airbean.l_code eq 'NPT' }">도쿄</c:when>
		<c:when test="${airbean.l_code eq 'KIX' }">오사카</c:when>
		<c:when test="${airbean.l_code eq 'FUK' }">후쿠오카</c:when>
		<c:when test="${airbean.l_code eq 'HKG' }">홍콩</c:when>
		<c:when test="${airbean.l_code eq 'BKK' }">방콕</c:when>
		<c:when test="${airbean.l_code eq 'BKI' }">코타키나발루</c:when>
		<c:when test="${airbean.l_code eq 'WO' }">블라디보스토크</c:when>
		<c:when test="${airbean.l_code eq 'JFK' }">뉴욕</c:when>
		<c:otherwise> ... </c:otherwise>
		</c:choose>
		(${airbean.l_code })&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;            
		</td>
		<td>${airbean.o_sdate }</td>
	</tr>
	</table>
	


	
	<br> &nbsp;
	<table>	
	<tr>
		<th>Economy</th><th>Business</th><th>First</th><th>출도착시간</th><th>항공편명</th>
	</tr>
	
	<!-- -------------  편도 -->
	<c:forEach var="a" items="${airinfo }">

	<c:set var="dr">${fn:substring(a.air_name,4,5) % 2}</c:set>
	<c:if test="${dr eq 1 }">
	
		<tr>

			
			<fmt:parseDate var="date_d" value="${a.o_stime }" pattern="HH:mm" />
			<fmt:parseNumber var="date_n" value="${date_d.time + 32400000}" integerOnly="true" />
			<fmt:parseNumber var="date_s" value="${a.o_soyo * 60000}" integerOnly="true" />
			<fmt:parseNumber var="format_hh" value="${(date_n + date_s)/3600000 }" integerOnly="true"/>
			<fmt:parseNumber var="format_mm" value="${(date_n + date_s)%3600000/60000 }" integerOnly="true" />
			<c:choose>
			<c:when test="${format_hh >= 24}">
            <fmt:parseNumber var="soyo_hh" value="${a.o_soyo / 60}" integerOnly="true"/>
            <c:set var="alive" value="${a.o_stime } ~ ${format_hh - 24}:${format_mm } (+1일) <br> (소요시간 : ${soyo_hh }시간 ${a.o_soyo % 60}분)" />
			<c:set var="alive_time" value="${format_hh - 24}:${format_mm } (+1일)"></c:set>
			</c:when>
			
            <c:otherwise>
            <fmt:parseNumber var="soyo_hh" value="${a.o_soyo / 60}" integerOnly="true"/>
            <c:set var="alive" value="${a.o_stime } ~ ${format_hh}:${format_mm } <br> (소요시간 : ${soyo_hh }시간 ${a.o_soyo % 60}분)" />
			<c:set var="alive_time" value="${format_hh }:${format_mm }"></c:set>
            </c:otherwise>
         	</c:choose>
			<c:set var="soyo_time" value="${soyo_hh }시간 ${a.o_soyo % 60}분"></c:set>
			
			
			<td><input type="radio" name="radioTxt" value="${a.o_price }" onclick="javascript:gogo('${a.o_stime }','${alive_time }','${soyo_time }','${a.air_name }','E')"><fmt:formatNumber value="${a.o_price }" pattern="#,###" /> KRW</td>
			<td><input type="radio" name="radioTxt" value="${a.o_price + a.o_price * 0.2 }" onclick="javascript:gogo('${a.o_stime }','${alive_time }','${soyo_time }','${a.air_name }','B')"><fmt:formatNumber value="${a.o_price + a.o_price * 0.2 }" pattern="#,###" /> KRW</td>
			<td><input type="radio" name="radioTxt" value="${a.o_price + a.o_price * 0.9 }" onclick="javascript:gogo('${a.o_stime }','${alive_time }','${soyo_time }','${a.air_name }','F')"><fmt:formatNumber value="${a.o_price + a.o_price * 0.9 }" pattern="#,###" /> KRW</td>

			
			
			<td>${alive }</td>
			<td>${a.air_name }</td>

		</tr>
	</c:if>
	</c:forEach>
</table>
<br>

<div>
<br>
<table>
<tr>
	<td colspan="4" style="text-align: left" >성인 </td>



	<td style="text-align: right" >${airbean.people }명 </td>
</tr>
<tr>
	<td colspan="4" style="text-align: left" >항공운임 </td>
	<td style="text-align: right" id="ap"></td>
</tr>
<tr>
	<td colspan="4" style="text-align: left" >공항사용료</td>
	<td style="text-align: right" id="gong" ></td>
</tr>
<tr>
	<td colspan="4" style="text-align: left" >유류할증료</td>
	<td style="text-align: right" id="you" ></td>
</tr>
<tr>
</tr>
<tr>
	<td>예상 결제 금액 : </td>
	<td colspan="4" style="text-align: right; color: blue" id="tot"></td>
</tr>
</table>
</div>
<br>
<br>
<b style="color: red">▷ 항공권의 운임 및 잔여 좌석수는 실시간 변동될 수 있습니다.</b><br>
▷ 해당 항공 스케줄은 정부인가 조건에 따라 예고없이 변경될 수 있습니다.<br>
<div>
<br>
<pre>
▣ 유의사항
☞ 이용안내 유류할증료와 공항시설사용료 및 기타수수료는 환율 및 발권일에 따라 변동될 수 있습니다.
☞ 공항시설사용료 및 각종 세금은 노선에 따라 별도 부과될 수 있습니다.
☞ 소아·유아 운임은 홈페이지 - 서비스 안내 - 항공권서비스 - 국제선 운임/국내선 운임 에서 확인하실 수 있습니다.
☞ 무료 기내휴대수하물 허용량은 7kg이며, 자세한 사항은 홈페이지 서비스안내-공항서비스-수하물서비스 에서 확인하실 수 있습니다.
☞ 왕복 항공권 구매 후 여정변경 시 가는 날이 오는 날보다 먼저 이어야 합니다.
☞ 타 항공사로 환승하실 경우, 해당 공항에서 위탁수하물을 수취한 후 다시 출입국 수속을 진행하여 주시기 바랍니다.
</pre>
</div>
</div>


<form action="complete" method="post">
<input type="hidden" name="l_code" value="${airbean.l_code }">
<input type="hidden" name="o_sdate" value="${airbean.o_sdate }">
<input type="hidden" name="people" value="${airbean.people }">
<input type="hidden" name="pay">

<input type="hidden" name="start">
<input type="hidden" name="end">
<input type="hidden" name="time">
<input type="hidden" name="a_name">
<input type="hidden" name="grade">
<input type="hidden" name="maxT" value="${num+1}">
<input type="hidden" name="g_id" value="${id}">

</form>
<input type="button" id="btnComp" value="예약하기">
<a href="goindex"><input type="button" id="redirect:/index.jsp" value="취소" ></a>

</body>
</html>

<!-- 예약후 확인 -->
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@taglib prefix="fmt" uri="http://java.sun.com/jsp/jstl/fmt" %>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
<script type="text/javascript" src="http://code.jquery.com/jquery-3.2.0.min.js" ></script>
<link rel="stylesheet" href="resources/css/detail.css">
<script type="text/javascript">
$(document).ready(function(){
	$('#btnMyBook').click(function(){
		
		location.href = "mybook?g_id=${id}";
		
	});
	
	$('#btnhome').click(function(){
		location.href = "airinfo";		
	});
	
});


</script>
</head>
<body>
<h1>${id}님 예약이 완료되었습니다!!!!!!!!!!!!!</h1>


<table>
	<tr>
		<th>출발지</th><th>도착지</th><th>출발날짜</th><th>출도착시간</th><th>편명</th><th>좌석클래스</th><th>결제금액</th>
	</tr>

	<tr>
		<td>인천 (ICN)</td>
		<td>
		<c:choose>
		<c:when test="${complete.l_code eq 'CJU' }">제주</c:when>
		<c:when test="${complete.l_code eq 'NPT' }">도쿄</c:when>
		<c:when test="${complete.l_code eq 'KIX' }">오사카</c:when>
		<c:when test="${complete.l_code eq 'FUK' }">후쿠오카</c:when>
		<c:when test="${complete.l_code eq 'HKG' }">홍콩</c:when>
		<c:when test="${complete.l_code eq 'BKK' }">방콕</c:when>
		<c:when test="${complete.l_code eq 'BKI' }">코타키나발루</c:when>
		<c:when test="${complete.l_code eq 'WO' }">블라디보스토크</c:when>
		<c:when test="${complete.l_code eq 'JFK' }">뉴욕</c:when>
		<c:otherwise> ... </c:otherwise>
		</c:choose>(${complete.l_code })
		</td>
		<td>${complete.o_sdate }</td>
		<td>${complete.start } ~ ${complete.end }<br>(${complete.time })</td>
		<td>${complete.a_name }</td>
		<td>
		<c:choose>
		<c:when test="${complete.grade eq 'E' }">Economy(일반석)</c:when>
		<c:when test="${complete.grade eq 'B' }">Business(이등석)</c:when>
		<c:when test="${complete.grade eq 'F' }">First(일등석)</c:when>
		</c:choose>
		</td>
		<td>
		<fmt:formatNumber value="${complete.pay }" pattern="#,###" /> KRW
		</td>
	</tr>
</table>
<br>
<br>
<input type="button" value="나의예약정보" id="btnMyBook">

<input type="button" value="확인" id="btnhome">

</body>
</html>
```
<br>
티켓을 프린트하기 위해서 다음과 같은 구성을 하였다.  
프린트를 위한 JavaScript  
```js
function pagePrintPreview(){
    var browser = navigator.userAgent.toLowerCase();
    if ( -1 != browser.indexOf('chrome') ){
               window.print();
    }else if ( -1 != browser.indexOf('trident') ){
               try{
                        //참고로 IE 5.5 이상에서만 동작함
                        //웹 브라우저 컨트롤 생성
                        var webBrowser = '<OBJECT ID="previewWeb" WIDTH=0 HEIGHT=0 CLASSID="CLSID:8856F961-340A-11D0-A96B-00C04FD705A2"></OBJECT>';
                        //웹 페이지에 객체 삽입
                        document.body.insertAdjacentHTML('beforeEnd', webBrowser);
                        //ExexWB 메쏘드 실행 (7 : 미리보기 , 8 : 페이지 설정 , 6 : 인쇄하기(대화상자))
                        previewWeb.ExecWB(7, 1);
                        //객체 해제
                        previewWeb.outerHTML = "./resources/param.html";
               }catch (e) {
                        alert("- 도구 > 인터넷 옵션 > 보안 탭 > 신뢰할 수 있는 사이트 선택\n   1. 사이트 버튼 클릭 > 사이트 추가\n   2. 사용자 지정 수준 클릭 > 스크립팅하기 안전하지 않은 것으로 표시된 ActiveX 컨트롤 (사용)으로 체크\n\n※ 위 설정은 프린트 기능을 사용하기 위함임");
               }
            
    }
    
}
```
티켓 구입  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/1e95520d75b448b9af39fc6ec07d009b" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>
티켓 구입내역 프린트의 경우 Google Loom에서 촬영되지 않아 사진으로 첨부  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Book2.PNG" height="100%" width="100%" /></div><br>


###  체크인
체크인이란 예약 몇일전에 사용자가 직접 좌석을 고를 수 있는 System이다.  
Class(일반, 비지니스, VIP)안에서 좌석을 고를 수 있으면 체크인을 하게 되면 티켓을 확인하여 Print하거나 E-mail로 발송하여 티켓을 확인할 수 있게 구성하였다.  

```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
    <%@taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
    <%@ taglib uri = "http://java.sun.com/jsp/jstl/functions" prefix = "fn" %>
    <%
    String g_id=(String) session.getAttribute("id"); 
    /*
    int inwon=2;
    String a_seat="testing105a";
    */
    %>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Bom Air : Seat Choice</title>
<!-- jQuery CDN -->
<script src="https://code.jquery.com/jquery-3.4.0.js"
  integrity="sha256-DYZMCC8HTC+QDr5QNaIcfR7VSPtcISykd+6eSmBW5qo=" crossorigin="anonymous"></script>
<!-- 합쳐지고 최소화된 최신 자바스크립트 -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/js/bootstrap.min.js"></script>
<!-- 합쳐지고 최소화된 최신 CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap.min.css">
<!-- 부가적인 테마 
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.2/css/bootstrap-theme.min.css">-->
<!-- <script src="https://code.jquery.com/jquery-3.3.1.min.js"></script> -->
<!-- 직접 정의한 CSS -->
<link rel="stylesheet" href="resources/css/seat.css"> 
<script>
var confirmValue=false;
var inwon=${inwon};
//console.log(inwon);

$(document).ready(function(){
	
	<c:forEach var="s" items="${data}">	
		<c:set var="seatNo" value="${fn:substring(s.s_no, 0, 1)}"></c:set>
		// console.log(${s.s_no});
	    $("#${s.s_no}_lb").attr("disabled",true)
	    $("#${seatNo}_div").attr("style", "background-color: red")
	</c:forEach>
	
	// 체크박스 클릭 시 
	$("input[type=checkbox]").change(function(){
		
		var seatNo=$(this).attr('id');
		
		if($(this).is(":checked")){
			if(inwon>0){ // 정상적인 상태
				$("#gogekSeat").val(seatNo);
	    		$("#modal_seatCheck").modal('toggle');
			} else { // 예약인원수 이상 클릭 시도할 때
				jQuery("label[for="+$(this).attr('id')+"]").addClass("active");
				alert("예약한 인원수만큼 좌석선택이 완료되었습니다." );
			} 
			
        } else{ // 이미 선택된 좌석을 한번 더 클릭 시
        	if(confirm("정말 해당 좌석 선택을 취소하시겠습니까?")){
        		var class_seatNo="."+seatNo;
        		console.log(class_seatNo);
        		jQuery(class_seatNo).remove();
        		jQuery("label[for="+$(this).attr('id')+"]").addClass("active");
        		inwon++;
        		alert(seatNo+"번 좌석을 취소하였습니다.");
        	} else {
        		jQuery("label[for="+$(this).attr('id')+"]").removeClass("active");
        	}
        }
		
		// 모달 상태에서 체크인버튼 클릭 시
		$("#btnCheckIn").click(function(){
			if($("#gogekName").val()===""){
				jQuery("label[for=gogekName]").empty();
				jQuery("label[for=gogekName]").append("탑승자 성명을 기입해주세요.");
				jQuery("label[for=gogekName]").fadeIn('fast');
				$("#gogekName").focus();
				setTimeout(function() {
					jQuery("label[for=gogekName]").fadeOut();
				}, 500);
				
			} else if($("#gogekBookNo").val()===""){
				jQuery("label[for=gogekBookNo]").empty();
				jQuery("label[for=gogekBookNo]").append("탑승자 예약번호를 기입해주세요.");
				jQuery("label[for=gogekBookNo]").fadeIn('fast');
				$("#gogekBookNo").focus();
				setTimeout(function() {
					jQuery("label[for=gogekBookNo]").fadeOut();
				}, 500);
				
			} else if($("#gogekPassport").val()===""){
				jQuery("label[for=gogekPassport]").empty();
				jQuery("label[for=gogekPassport]").append("탑승자 여권번호를 기입해주세요.");
				jQuery("label[for=gogekPassport]").fadeIn('fast');
				$("#gogekPassport").focus();
				setTimeout(function() {
					jQuery("label[for=gogekPassport]").fadeOut();
				}, 500);
				
			} else {
				var tagForSeatCheck="";
				tagForSeatCheck+="<label class='"+$("#gogekSeat").val()+"'>";
				tagForSeatCheck+="<input type='hidden' name= 's_no' value='"+$("#gogekSeat").val()+"'>";
				tagForSeatCheck+="<input type='hidden' name= 't_no' value='"+$("#gogekBookNo").val()+"'>";
				tagForSeatCheck+="<input type='hidden' name= 'b_name' value='"+$("#gogekName").val()+"'>";
				tagForSeatCheck+="<input type='hidden' name= 'b_pp' value='"+$("#gogekPassport").val()+"'>";
				tagForSeatCheck+="<input type='hidden' name= 'g_id' value='"+"<%=g_id%>"+"'>";
				tagForSeatCheck+="</label>";
				jQuery(".hiddenArea").append(tagForSeatCheck);
				
				var tagForConfirm="";
				tagForConfirm+="<label class='"+$("#gogekSeat").val()+"'>";
				tagForConfirm+="좌석번호: "+$("#gogekSeat").val()+"&nbsp;&nbsp;";
				tagForConfirm+="탑승자명: "+$("#gogekName").val()+"&nbsp;&nbsp;";
				tagForConfirm+="여권번호: "+$("#gogekPassport").val();
				tagForConfirm+="</label><br><br>";
				jQuery("#modal_confirmBody").append(tagForConfirm);
				
				console.log($("#gogekSeat").val()+", "+$("#gogekBookNo").val()+", "+$("#gogekName").val()+", "+$("#gogekPassport").val()+", "+"<%=g_id%>");
				
				inwon--;
				alert($("#gogekSeat").val()+"번 좌석 선택이 완료되었습니다.");
				$("#modal_seatCheck").modal('toggle');
				
				$("#gogekName").val("");
				$("#gogekPassport").val("");
			}
		})
		
		$("#choiceCancel").click(function(){
			console.log($("#gogekSeat").val()+"취소");
			$("#gogekName").val("");
			$("#gogekPassport").val("");
			jQuery("label[for="+$("#gogekSeat").val()+"]").removeClass("active");
		});
		
		$("#btnGoNext").click(function(){
			$("#modal_confirm").modal('toggle');
			$("form[name=frm]").submit();
		});
		
		$("#btnBack").click(function(){
			// alert("go back");
			$("#modal_confirm").modal('toggle');
		});
		
    });
});

function confirmCheck(){
	if(inwon===0){
		console.log(inwon);
		$("#modal_confirm").modal('toggle');
	}else{
		alert("예약한 인원만큼 좌석을 체크인 해주세요.");				
	}
}
</script> 
</head>
<body> 

<div class="container basicFont">
	<div class="SeatTop">
		<a href="index.jsp"><img src="resources/images/bomair_logo.png"/></a>
		<p class="font_title">좌석 지정 페이지</p>
		<hr class="hrCss">
	</div>
	
	<!-- 좌석 현황 표출 -->
	<div class="showSeatArea">
			<div class="btn-group" data-toggle="buttons" id="F_div">
				<label class="btn btn-big btn-primary" for="F1" id="F1_lb"><input
					type="checkbox" autocomplete="off" id="F1" value="F1">F1</label> <label
					class="btn btn-big btn-primary" for="F2" id="F2_lb"><input
					type="checkbox" autocomplete="off" id="F2">F2</label> <label
					class="btn btn-big btn-primary" for="F3" id="F3_lb"><input
					type="checkbox" autocomplete="off" id="F3">F3</label>
			</div>
			<br>
			<br>
			<div class="btn-group" data-toggle="buttons" id="E_div">
				<label class="btn btn-big btn-primary" for="E1" id="E1_lb"><input
					type="checkbox" autocomplete="off" id="E1">E1</label> <label
					class="btn btn-big btn-primary" for="E2" id="E2_lb"><input
					type="checkbox" autocomplete="off" id="E2">E2</label> <label
					class="btn btn-big btn-primary" for="E3" id="E3_lb"><input
					type="checkbox" autocomplete="off" id="E3">E3</label>
			</div>
			<br>
			<br>
			<div class="btn-group" data-toggle="buttons" id="B_div">
				<label class="btn btn-big btn-primary" for="B1" id="B1_lb"><input
					type="checkbox" autocomplete="off" id="B1">B1</label> <label
					class="btn btn-big btn-primary" for="B2" id="B2_lb"><input
					type="checkbox" autocomplete="off" id="B2">B2</label> <label
					class="btn btn-big btn-primary" for="B3" id="B3_lb"><input
					type="checkbox" autocomplete="off" id="B3">B3</label>
			</div>
			<br>
			<br>
		</div>
	
	<!-- Modal : 좌석 체크  -->
	<div class="modal fade" id="modal_seatCheck" tabindex="-1" role="dialog" 
		aria-labelledby="myModalLabel" aria-hidden="true" data-backdrop="static" data-keyboard="false">
	  <div class="modal-dialog">
	    <div class="modal-content">
	      <div class="modal-header">
	        <h4 class="modal-title" id="myModalLabel">체크인 : 좌석선택</h4>
	      </div>
	      <div class="modal-body">
	      		<div class="form-group">
		            좌석번호 : <label for="gogekSeat" class="control-label"></label>
		            <input type="text" class="form-control" id="gogekSeat" value="" readonly/>
		          </div>
		           <div class="form-group">
		            탑승자 예약번호 : <label for="gogekBookNo" class="control-label dispAlert" style="color:red;"></label>
		            <input type="text" class="form-control" id="gogekBookNo" value="${t_no }" readonly/>
		          </div>
		          <div class="form-group">
		            탑승자 성명 : <label for="gogekName" class="control-label dispAlert" style="color:red;"></label>
		            <input type="text" class="form-control" id="gogekName">
		          </div>
		          <div class="form-group">
		            탑승자 여권번호 : <label for="gogekPassport" class="control-label dispAlert" style="color:red;"></label>
		            <input type="text" class="form-control" id="gogekPassport">
		          </div>
	      </div>
	      <div class="modal-footer">
	        <button class="btn btn-default" data-dismiss="modal" id="choiceCancel">취소</button>
	        <button class="btn btn-primary" id="btnCheckIn">체크인</button>
	      </div>
	    </div>
	  </div>
	</div>
	
	<!-- Modal : 체크인 내역 최종확인 -->
	<div class="modal fade" id="modal_confirm" tabindex="-1" role="dialog" 
		aria-labelledby="myModalLabel" aria-hidden="true" data-backdrop="static" data-keyboard="false">
	  <div class="modal-dialog">
	    <div class="modal-content">
	      <div class="modal-header">
	        <h4 class="modal-title" id="myModalLabel">좌석 선택 내역 확인</h4>
	        <h5 class="modal-title" id="myModalLabel">한 번 체크인한 좌석은 변경할 수 없으니 신중히 확인해주세요.</h5>
	      </div>
	      <div id="modal_confirmBody" class="modal-body">
	      
	      </div>
	      <div class="modal-footer">
	        <button class="btn btn-default" data-dismiss="modal" id="btnBack">돌아가기</button>
	        <button class="btn btn-primary" id="btnGoNext">최종 완료하기</button>
	      </div>
	    </div>
	  </div>
	</div>
	
	<!-- 최종전송 폼 -->
	<form action="showCheckinInfo" method="POST" name="frm">
		<input type="hidden" name="seatTableName" value="${a_seat }">
		<div class="hiddenArea"></div>
		<div class="css_btnSubmit">
			<input type="button" value="체크인 완료하기" class="btn btn-big btn-success" onclick="confirmCheck();">
		</div>
	</form>

	<br><br>	
	<!-- <a href="showCheckinInfo">Go to 체크인 내역 확인 페이지 (확인용, 철거예정)</a> -->
	
</div>	
</body>
</html>
```
<br>
체크인을 하고 나서는 티켓 내역에서 Ticket을 Mail로 발송받거나 프린트하여 사용할 수 있다.  

체크인 과정  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/d4d464ea395e4e88859415f390486b93" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>
Ticket Print
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Book3.PNG" height="100%" width="100%" /></div><br>
<br>
Ticket Mail 발송
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Book4.PNG" height="100%" width="100%" /></div><br>
<br>
<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
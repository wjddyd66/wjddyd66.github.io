---
layout: post
title:  "Spring-Project-MainPage"
date:   2019-06-27 06:30:00 +0700
categories: [Project]
---

###  Spring-Project-MainPage
Project의 Main Page이다.  
Main Page의 경우 Header, Body, Bottom으로 구성하여 계속해서 필요한 Header와 Bottom의 경우 계속해서 사용하도록 하였다.  
Design의 경우 BootStrap을 활용하여 꾸미게 되었고, 중요한 기능인 항공권 예매와 렌트카 예매가 한눈에 보이게 구성하였다.  
화면에 보이는 공지사항은 Ajax를 활용하여 정보를 가져오게 되고, Header의 경우 Session에 따라 특정 메뉴가 보이게 구성하였다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo5.PNG" height="100%" width="100%" /></div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Project/bomair_logo6.PNG" height="100%" width="100%" /></div>  
<br>

###  Session
Session을 통하여 권한을 부여하기 위하여 Controller에서 session을 설정하는 방법이다.  
id라는 이름으로 User가 Login한 userInputId의 값을 저장하고 MainPage인 index.jsp로 돌아가게 된다.  
아래 코드는 Session을 설정하는 방법이다.  
```java
	@RequestMapping(value="login", method=RequestMethod.POST)
	public ModelAndView submitLogin(HttpSession session, @RequestParam("g_id") String userInputId, @RequestParam("g_pwd") String userInputPwd) {
		ModelAndView view=new ModelAndView();
		if(inter.loginCheck(userInputId, userInputPwd)) {
			session.setAttribute("id", userInputId);
			view.setViewName("redirect:/index.jsp");
		} else {
			view.setViewName("login");
			view.addObject("state", "loginFail");
		}
		return view;
	}
```
<br>
Session에 ID값을 저장한 뒤 여러 군데에서 가져와서 사용이 가능하게 된다.  
아래 코드는 Header(Navigation Bar)에서 id의 값을 비교하고 id의 값에 따라서 Header의 값이 다르게 보이게 하는 코드이다.  
<span style ="color: red">**id의 값이 admin이면 관리자의 특정 Menu가 보이게 설정하였다. 즉, DB에 admin이라는 계정을 만들어야 한다는 한계가 존재하게 된다.**</span><br>
```jsp
<!-- Controller에서 설정한 Session값 가져오기 -->
<%
	String id = (String) session.getAttribute("id");
%>

<!-- 가져온 Session값으로 Header Menu 변하게 하기 -->
<c:if test="${id ne null }">
					<li class="nav-item dropdown" style="color:white;">${id }님<br>안녕하세요!</li>				
				</c:if>
				
				<c:if test="${id eq 'admin' }">
					<li class="nav-item dropdown">
						<a class="nav-link dropdown-toggle" href="#" id="dropdown01"
					data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Admin Page</a>
					<div class="dropdown-menu" aria-labelledby="dropdown01">
						<a class="dropdown-item" href="insert">데이터베이스</a> 
						<a class="dropdown-item" href="cal">매출현황 </a>
					</div></li>
				</c:if>
```
<br>
위와 같이 관리자의 권한이 필요한 곳에서 계속하여 Session의 값을 비교하여 권한을 부여하였다.  
Header-Login X  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Login2.PNG" height="150" width="600" /></div><br>
Header-Login O  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Login.PNG" height="150" width="600" /></div><br>
Header-Admin  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Login-Amin.PNG" height="150" width="600" /></div><br>  

###  빠른구매 창
가장 Main인 기능이고 사람들이 주로 사용하는 목적인 예약을 필요한 정보를 입력하여 빨리 할 수 있게 Main Page에 가장 잘 보이는 곳에 설정하였다.  
항공권의 경우 편도와 왕복일 경우 다른 Menu가 나오게 되고 Rent카를 누르게 되었을 경우도 구매 창이 바뀌게 구성하였다.  
div의 같은 위치에서 show 와 hide를 사용하여 구성하게 되었다.  
날짜 선택의 경우 보기 편하게 달력 UI와 JS를 사용하여 구성하게 되었다.(도착일이 출발일 보다 빠를수 없는 예외처리적용)  
```js
/* 항공권, 렌트카에 따라 테이블 보이기 */
			$("input[name='reservation']").click(function() {
				$("#reservation-table").show();
				$("#rent-table").hide();
				$("#searchHang").show();
				$("#searchRent").hide();

			});

			$("input[name='rent']").click(function() {
				$("#rent-table").show();
				$("#reservation-table").hide();
				$("#searchRent").show();
				$("#searchHang").hide();
			});

			$("input[name='flightWay']").click(function() {
				if ($("input[name='flightWay']:checked").val() == 'round') {
					$("#backDate").show();
					$("#goDate").css('width', '50%');
					$("#goDate").attr('colspan', '1');
				} else {
					$("#backDate").hide();
					$("#goDate").css('width', '100%');
					$("#goDate").attr('colspan', '2');
				}

			});
			
			/* 날짜 선택 */
			$("#GoDateChoose").datepicker(
					{
						changeMonth : true,
						changeYear : true,
						nextText : '다음 달',
						prevText : '이전 달',
						showButtonPanel : true,
						currentText : '출발 날짜',
						closeText : '닫기',
						dateFormat : "yymmdd",
						minDate : -0,
						maxDate : 1000,
						dayNamesMin : [ '월', '화', '수', '목', '금', '토', '일' ],
						monthNamesShort : [ '1', '2', '3', '4', '5', '6', '7',
								'8', '9', '10', '11', '12' ],
						monthNames : [ '1월', '2월', '3월', '4월', '5월', '6월',
								'7월', '8월', '9월', '10월', '11월', '12월' ]

					});
```
<br>
<span style ="color: red">**Google Loom을 사용하여 촬영하였으나, Alert창은 보이지 않는 문제가 있다. 출발일이 도착일보다 빠르면 Alert창이 보이게 구성하였다.**</span><br>
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/6e9d22a37ae1476bbfac03abbcb3c972" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

###  공지사항
공지사항인 경우 Ajax로서 요청을 하였다. 또한 Main Page의 Design유지를 위하여 내용이 너무 길면 내용을 잘라서 가져오는 것으로 구성하였다. (제목은 짧다고 가정하에 내용만 예외처리로 생각하였다.)  
공지사항 Ajax 통신 JavaScript  
```js
/*공지사항 가져오기 Ajax 통신*/
$(document).ready(
		function() {
			$.ajax({
				type : "get",
				url : "gong_main",
				dataType : "json",
				success : function(data) {
					var str="";
					for(var i=0;i<data.length;i++){
						str+="<li><a href='gong_detail?num="+data[i].num+"&spage=1' class='float-left'>"
							+ data[i].title + "</a><br><p>" + data[i].con
							+ "</p></li>"
					}
					$("#gonggi_content").html(str);
				},
				error : function() {
					alert("에러발생");
				}
			});
```
<br>
공지사항 내용일 길 경우 요약하는 Controller
```java
	//내용이 길 경우 요약하기
	@RequestMapping("gong_main")
	@ResponseBody
	public List<gongDto> getJson() {
		List<gongDto> list = inter.selectMain();
		ArrayList<gongDto> list2 = new ArrayList<gongDto>();
		int k = 0;
		try {
			while (k < list.size()) {
				gongDto dto = new gongDto();
				dto.setTitle(list.get(k).getTitle());
				dto.setNum(list.get(k).getNum());
				if (list.get(k).getCon().length() > 50)
					dto.setCon(list.get(k).getCon().substring(0, 48) + "...");
				else
					dto.setCon(list.get(k).getCon());
				list2.add(dto);
				k++;
			}
		} catch (Exception e) {

		}
		return list2;
	}
```
<br>
MainPage 공지사항  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Gong.PNG" height="250" width="600" /></div><br>
실제 DB 공지사항  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Gong2.PNG" height="250" width="600" /></div><br>
<br>
<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
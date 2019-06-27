---
layout: post
title:  "Spring-Project-공지사항"
date:   2019-06-27 07:30:00 +0700
categories: [Spring]
---

###  공지사항-DB 구성
공지사항을 위한 DB구성은 아래와 같다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE1.PNG" height="100%" width="100%" /></div><br>

###  공지사항-Pgae 구성
공지사항의 Page는 크게 Header, Body, Bottom으로 구분되어있다.  
Header 와 Bottom 같은경우 Main Page와 같은 구성으로 되어있다.  
공지사항의 Page의 Body같은 경우 Session의 id값을 비교하여 id가 admin 즉, 관리자 이면 새글쓰기, 기존 글 수정, 삭제가 보이도록 구성하였다.  
Session의 id값을 비교하여 관리자를 알아내는 것은 아래 코드로 구현을 하였다.  

```jsp
		<!-- 게시글에 관리자 권한 에따라 목록 보이기 -->
		<table border="1" class="table" style="margin-top: 100px">
			<tr style="background-color: silver; color: black;">
				<th>번호</th>
				<th>제목</th>
				<th>조회수</th>
				<th>날짜</th>
				<c:if test="${id eq 'admin' }">
					<th colspan="2">관리자 권한</th>
				</c:if>
			</tr>
			<c:forEach var="s" items="${list}">
				<tr>
					<td>${s.num }</td>
					<td
						onClick="location.href='gong_detail?num=${s.num}&spage=<%=request.getParameter("spage")%>&sword=${sword}'"
						style="cursor: pointer; width: 500px;">${s.title }</td>
					<td>${s.readcnt }</td>
					<td>${s.bdate }</td>
					<c:if test="${id eq 'admin' }">
						<td><a
							href="gong_update?num=${s.num}&spage=<%=request.getParameter("spage")%>&sword=${sword}">수정</a></td>
						<td><a href="#" onclick="delchk(${s.num },'${sword}')">삭제</a></td>
					</c:if>
				</tr>
			</c:forEach>
		</table>

<!-- 게시글에 관리자 권한 에따라 새글 쓰기 추가 -->
		<div style="text-align: center; margin-bottom: 120px;">
			<form action="gong_list?spage=1" name="frm" method="post">
				<input type="text" name="sword"> <input type="submit"
					class="btn btn-outline-dark" value="검색" id="btnSearch">
				<c:if test="${id eq 'admin' }">
					<a class='nav-link' href='gong_write'>새글 추가</a>
				</c:if>
			</form>
		</div>
	</section>
```
일반 User 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE2.PNG" height="250" width="600" /></div><br>
<br>
Admin 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE4.PNG" height="250" width="600" /></div><br>
<br>

###  회원가입 정규성 검사
회원가입의 경우 id는 DB에서 Primary Key로 선언하여 실제로 가입되어있는 회원이있나 검사하고 확인하여야 했다.  
현재 상태의 정보를 그대로 가지고 있어야 하기 때문에(ID말고 다른것을 입력 뿐만 아니라 1페이지에서 약관동의를 한 정보도 가지고 있어야 한다.) Ajax로 통신하여 DB에서 ID가 겹치는 것이 있으면 메세지로서 바로 확인가능하도록 구성하였다.  
```js
var deny = 't';
$(document).ready(function() {
	$("#idcheck").on("click", function() {
		$.ajax({
			type : "get",
			url : "id_check",
			data : {
				"id" : $("#id").val()
			},
			dataType : "json",
			success : function(data) {
				if (data.Check === "success") {
					$("#id_span_fail").hide();
					$("#id_span").text("사용가능한 아이디입니다.");
					$("#id_span").show();

				} else if (data.Check === "fail") {
					$("#id_span").hide();
					$("#id_span_fail").text("아이디가 중복되었습니다.");
					$("#id_span_fail").show();
				}

				if (data.Check2 === "deny") {
					deny = 'f';
				}
			},
			error : function() {
				alert("에러발생");
			}
		});
	});
});
```
<br>
중복된 ID가 있는경우  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/ID3.PNG" height="250" width="600" /></div><br>
<br>
###  회원가입 고려 사항(쿠키)
Chrome이나 Internet Explor 사용시 사용자의 정보가 Cookie에 남아 Local 저장공간에 남게 된다.  
이러한 경우 Browser에서 Login을 편하게 하기 위하여 ID와 비밀번호가 자동으로 입력되는 기능을 제공하게 된다.  
회원가입창에도 특정 Text입력란에 가장 최근에 Login한 정보가 남게되어 다음 Code를 추가하게 되어 해결하였다.  
```jsp
			<!--remove autocomplete-->
			<input style="display: none" aria-hidden="true"> 
			<input type="password" name='non_auto' value=' ' style="display: none" aria-hidden="true">
			<!--remove autocomplete end-->
			<!--real input start-->
			<input type="text" name='non_auto' value=' ' autocomplete="false" required style="display: none"> 
			<input type="password" name="password" value=' ' autocomplete="new-password" style="display: none">
			<!--real input end-->
```
<br>
display:none으로 되어있어 사용자에게 보이지 않고 value=''으로서 아무 값을 주지 않아 사용자의 Login 정보가 담기게 되어도 다른 User가 보지 못하게 구성하였다.  
<br>
###  회원가입 주소
회원가입 거주지 같은경우 KaKao에서 제공하는 우편번호 시스템을 사용하였다.  
많은 양의 정보와 정확한 지리를 DB에 저장할 수 없는 상황이여서 이런 방법을 채택하게 되었다.  
우편번호 서비스:<http://postcode.map.daum.net/guide>  
이러한 KaKao 우편번호 서비스는 iframe에 띄워서 제공하는 방식으로 구성하였다.  
```js
//다음 API
function execDaumPostcode() {
	new daum.Postcode({
		oncomplete : function(data) {
			
			var addr = ''; // 주소 변수
			var extraAddr = ''; // 참고항목 변수

			if (data.userSelectedType === 'R') {
				addr = data.roadAddress;
			} else {
				addr = data.jibunAddress;
			}

			if (data.userSelectedType === 'R') {

				if (data.bname !== '' && /[동|로|가]$/g.test(data.bname)) {
					extraAddr += data.bname;
				}

				if (data.buildingName !== '' && data.apartment === 'Y') {
					extraAddr += (extraAddr !== '' ? ', ' + data.buildingName
							: data.buildingName);
				}

				if (extraAddr !== '') {
					extraAddr = ' (' + extraAddr + ')';
				}

				document.getElementById("extraAddress").value = extraAddr;

			} else {
				document.getElementById("extraAddress").value = '';
			}

			document.getElementById('postcode').value = data.zonecode;
			document.getElementById("address").value = addr;
			//추가
			document.getElementById("detailAddress").value='';
			
			document.getElementById("detailAddress").focus();

			element_layer.style.display = 'none';
		},
		width : '100%',
		height : '100%',
		maxSuggestItems : 5
	}).embed(element_layer);

	// iframe을 넣은 element를 보이게 한다.
	element_layer.style.display = 'block';
	initLayerPosition();
}

//다음 API 입력값 받아오기
function addradd() {
	if (document.getElementById("address").value !== null
			&& document.getElementById("detailAddress").value !== null) {
		document.getElementById("g_addr").value = document
				.getElementById("address").value
				+ document.getElementById("detailAddress").value;
		// alert(document.getElementById("g_addr").value);
	}
}
```
<br>
###  아이디, 비밀번호 찾기
아이디 같은경우 중요한 정보가 아니기 때문에 바로 보여지는 것이 가능하였으나, 비밀번호 같은 경우는 개인적인 정보이므로 쉽게 정보를 보여주어서는 안된다.  
따라서 회원가입되어있는 정보로 Mail을 발송하는 것으로 구성하였다.  
Mail은 G-mail을 통하여 발송하는 것으로 구성하였다.  
메일을 발송하기 위한 Dependency 추가
```xml
		<!-- Java Mail API -->
		<dependency>
			<groupId>javax.mail</groupId>
			<artifactId>mail</artifactId>
			<version>1.4.3</version>
		</dependency>

		<!-- smtp -->
		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-mail</artifactId>
			<version>2.0.1.RELEASE</version>
		</dependency>

		<dependency>
			<groupId>org.springframework</groupId>
			<artifactId>spring-context-support</artifactId>
			<version>3.2.8.RELEASE</version>
		</dependency>

		<dependency>
			<groupId>javax.mail</groupId>
			<artifactId>mail</artifactId>
			<version>1.4</version>
		</dependency>
```
<br>
Mail을 G-mail로 발송하기 위해서는 Google계정과 연동이 되어야 한다.  
이러한 Google 계정 연동을 위한 개인적인 정보는 mail.properties를 통하여 properties 파일에 저장하여 보안성을 강구하였다.  
```code
host=smtp.gmail.com
port=587
username=hwangjeongyong4@gmail.com
password=비밀번호
```
<br>
발송하는 Mail을 properties와 연동하고 내용을 추가하여 넣는 코드이다.  
비밀번호 같은경우 사용자가 입력한 내용과 일치하는 고객의 비밀번호와 Mail을 가져와서 발송하게 되었다.  
```java
	public boolean sendPwd(HashMap<String, Object> paramap) {
		boolean b = false;
		HashMap<String, Object> map = getMail_Pwd(paramap);
		if (map.size() != 0) {
			b = true;
			String setfrom = "hwangjeongyong4@gmail.com";
			String tomail = (String) map.get("g_mail"); // 받는 사람 이메일
			String title = "BOM AIR 비밀번호 찾기 기능 입니다."; // 제목
			try {
				MimeMessage message = mailSender.createMimeMessage();
				MimeMessageHelper messageHelper = new MimeMessageHelper(message, true, "UTF-8");
				messageHelper.setFrom(setfrom); // 보내는사람 생략하거나 하면 정상작동을 안함
				messageHelper.setTo(tomail); // 받는사람 이메일
				messageHelper.setSubject(title); // 메일제목은 생략이 가능하다
				String text = "고객님의 비밀번호는 " + (String) map.get("g_pwd");
				messageHelper.setText(text, true);
				mailSender.send(message);
			} catch (Exception e) {
				System.out.println(e);
			}

		}
		return b;
	}
```
<br>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/ID4.PNG" height="100%" width="100%" /></div><br>

<hr>
참조:<https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
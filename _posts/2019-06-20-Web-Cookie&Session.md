---
layout: post
title:  "Web-Cookie&Session"
date:   2019-06-20 09:00:00 +0700
categories: [Web]
---

###  Cookie&Session
Session
 - 클라이언트와 웹 서버간에 통신 연결에서 두 개체의 활성화된 접속을 의미한다.
 - 각각의 클라이언트마다 고유의 Session ID를 가지고 있다.

Cookie
 - 서버측에서 클라이언트 측에 상태 정보를 저장하고 추출할 수 있는 메커니즘
 - 쿠키는 FIFO구조로 인하여 시간이 지나면 사라질 수 있다.
 - 웹사이트에 접속하게 되었을때, 저장경로에 쿠키 파일을 전부 읽은 후 갱신할 내용이 있으면 갱신 및 생성하고, 갱신할 내용이 없다면 그대로 가져다가 사용한다.
 - 쿠키는 암호화 되어 저장되어 내용을 확인할 수 없다.

암호화된 쿠키 사진  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js45.JPG" height="150" width="600" /></div><br>

###  Html
a 링크를 통하여 Servlet 에게 CookieLogin 과 SessionLogin을 각각 요청하는 코드이다.  


```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Cookie_Session</title>
</head>

<body>
	<h1>Cookie</h1>
	<a href="CookieLogin">쿠키 연습</a>
	<br>
	<h1>Session</h1>
	<a href="SessionLogin">세션 연습</a>

</body>
</html>
```
###  CookieLogin
CookieLogin 요청이 들어왔을때 수행하는 곳이다.  
doGet(): 처음 a링크를 타고 들어왔을때 수행하는 곳이다.
 -  클라이언트의 모든 쿠키를 읽는다.
 -  클라이언트의 쿠키에서 필요한 정보를 Decode하여 가져온다.
 -  Cookie가 없으면: Login Page로가서 정보를 작성하고 doPost()로값을 보낸다.
 -  Cookie가 있으면 Cookie에서 Decode한 값을 보여준다.

doPost(): 처음 a링크를 타고 들어왔을때 수행하는 곳이다.
 -  요청받는 정보의 ID와Passward가 각각 kbs, 123 인지 확인한다.
 -  위조건을 만족하는 경우 쿠키 유효기간을 1년으로 하여 저장한다.

```java
package pack;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.URLDecoder;
import java.net.URLEncoder;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/CookieLogin")
public class CookieLogin extends HttpServlet {
	
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		response.setContentType("text/html; charset=utf-8");
		PrintWriter out=response.getWriter();
		out.println("<html><body>");
		String id=null;
		String pwd=null;
		
		// 쿠키를 만들때는 try-catch 사용
		try {
			Cookie[] cookies=request.getCookies(); // 클라이언트의 모든 쿠키 읽기
			for (int i = 0; i < cookies.length; i++) {
				String name=cookies[i].getName();

				// 쿠키찾기
				if(name.equals("id")) { 
					id=URLDecoder.decode(cookies[i].getValue(), "utf-8");
				}
				if(name.equals("pwd")) {
					pwd=URLDecoder.decode(cookies[i].getValue(), "utf-8");
				}
			}
		} catch (Exception e) {
			
		}
		
		if(id!=null && pwd!=null) {
			out.println("Cookie에 저장된 ID:"+id+"<br>");
			out.println("Cookie에 저장된 Passward:"+pwd);
			out.println("<html><body>");
			out.close();
			return;
		}
		
		out.println(" * 로그인 * ");
		out.println("<form method='post'>");
		out.println("id: <input type='text' name='id'><br>");
		out.println("pwd: <input type='text' name='pwd'><br>");
		out.println("<input type='submit' value='전송'>");
		out.println("</form></body></html>");
		
		out.close();
	}

	protected void doPost(HttpServletRequest request, HttpServletResponse response) 
    throws ServletException, IOException {
		request.setCharacterEncoding("utf-8");
		response.setContentType("text/html; charset=utf-8");
		PrintWriter out=response.getWriter();
		
		String id=request.getParameter("id");
		String pwd=request.getParameter("pwd");
		
		
		//임의로 id와 비밀번호가 kbs,123이라 가정한다.
		if(id.equals("kbs")&&pwd.equals("123")) {
			// 쿠키 제작시에는 트라이캐치를!
			try {
				id=URLEncoder.encode(id, "utf-8"); // 암호화
				Cookie idCookie=new Cookie("id", id);
				idCookie.setMaxAge(60 * 60 * 24 * 365); // 쿠키 유효기간: 1년 
				
				pwd=URLEncoder.encode(pwd, "utf-8"); // 암호화
				Cookie pwdCookie=new Cookie("pwd", pwd);
				pwdCookie.setMaxAge(60 * 60 * 24 * 365); // 쿠키 유효기간: 1년 
				
				response.addCookie(idCookie); // 클라이언트의 pc에 저장
				response.addCookie(pwdCookie);
				out.println("로그인 성공: 쿠키 작성됨");
			} catch (Exception e) {}
		} else {
			out.println("로그인 실패");
		}
		out.println("</body></html>");
		out.close();
		
	}
}

```
<br>
결과
1. 처음 쿠키가 없는 경우 로그인 작성이 나오게 된다.
2. 로그인 작성후 쿠기가 저장되는 것을 알 수 있다.
3. 쿠키를 저장하고 난 뒤에는 바로 로그인이 되는 것을 확인할 수 있다.  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/fd5b666ed72d4c2ca52ffcbb36473f29" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

###  SessionLogin
1. Session이 없으면
 - 세션의 지속시간 10초로 설정(기본값은 30분)
 - 세션에 "name"="kbs" 저장하기 
 - 세션에 "pwd"="123" 저장하기
2. Session이 있으면: Session에 저장된 값 보여주기

```java
package pack;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

/**
 * Servlet implementation class SessionTest
 */
@WebServlet("/SessionLogin")
public class SessionLogin extends HttpServlet {
	protected void service(HttpServletRequest request, HttpServletResponse response) 
    throws ServletException, IOException {
		HttpSession session=request.getSession(true); // 세션이 있으면 읽고, 없으면 생성
		
		session.setMaxInactiveInterval(10); // 10초동안 유효. 기본값은 30분.
		
		if(session!=null) {
			session.setAttribute("name", "kbs"); // 세션에 "name"="kbs" 저장하기
			session.setAttribute("pwd", "123"); // 세션에 "pwd"="123" 저장하기
			
			request.setCharacterEncoding("utf-8");
			response.setContentType("text/html; charset=utf-8");
			PrintWriter out=response.getWriter();
			out.println("<html><body>");
			out.println("Session id: "+session.getId());
			out.println("<br> Session에 저장된 ID: "+session.getAttribute("name"));
			out.println("<br> Session에 저장된 Passward:: "+session.getAttribute("pwd"));
			out.println("</body></html>");
			out.close();
		}
	}

}
```
<br>
결과
1. 접속후 10초간 기다리다 새로고침 한 결과 새로운 Session이 할당되는 것을 확인 할 수 있다.
2. 접속후 10초 안에 새로고침 한 결과 같은 Session이 지속되는 것을 확인 할 수 있다.  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/ebb408944dae4cc199b534d1c30bedba" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div><br>

<hr>
참조:<a href="https://github.com/wjddyd66/Web/tree/master/Session_Cookie">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
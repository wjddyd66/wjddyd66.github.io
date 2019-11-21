---
layout: post
title:  "Web-Forward&Redirect"
date:   2019-06-21 08:00:00 +0700
categories: [Web]
---

###  Forward&Redirect
페이지의 요청을 처리하는 방식이다.  

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table" style="width:100%">
	<tbody>
	<tr>
		<td></td><td>Forward</td><td>Redirect</td>
	</tr>
	<tr>
		<td>대상</td><td>Server to Server</td><td>Client to Server</td>
	</tr>
		<tr>
		<td>Data Input</td><td>request.setAttribute</td><td>URL</td>
	</tr>
		<tr>
		<td>Data Output</td><td>request.getAttribute</td><td>request.getParameter</td>
	</tr>

	<tr>
		<td>사용이유</td><td>객체 재상용, 공유</td><td>URL 변화</td>
	</tr>
	</tbody>
</table>
<br>

###  Html
각각 Forward 와 Redirect라는 URL로 요청을 보내게 된다.   


```html
<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8">
    <title>Insert title here</title>
</head>

<body>
	Foward 방법: <br>
    <form action="Forward" method="post">
        Information Input: <input type="text" name="data" value="tom">
        <input type="submit">
    </form><br>
    
    Redirect 방법: <br>
    <form action="Redirect" method="post">
        Information Input: <input type="text" name="data" value="tom">
        <input type="submit">
    </form>

</body>

</html>
```
<br>
실행 화면:
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/FR.JPG" height="150" width="700" /></div>
<br>
###  Forward
Forward방식으로 요청을 처리하는 방식이다.  
1) Forward.java
 - Forward라는 요청이 들어왔을때 요청하는 곳이다. 
 - name변수에 HTML에서 data의 이름을 가진 변수의 값을 저장한다.
 - data변수에 "Foward"와 name의 변수를 담는다.
 - Forward.jsp에 data를 전달한다.

```java
package pack;

import java.io.IOException;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class Forward
 */
@WebServlet("/Forward")
public class Forward extends HttpServlet {
	protected void service(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		request.setCharacterEncoding("utf-8");
		String name = request.getParameter("data");
		String data[] = {"Forward",name};
		
		//forwarding - Server to Server
		request.setAttribute("data", data);
		RequestDispatcher dispatcher = request.getRequestDispatcher("Forward.jsp");
		dispatcher.forward(request, response);
		
	}

}

```

2) Forward.jsp
 - 받은 자료를 출력하는 곳 이다.

```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
	Called File by Servlet
	<br>
	<%
		request.setCharacterEncoding("utf-8");
		//redirect Method
		String[] data = (String[]) request.getAttribute("data");
		out.println("방식은"+data[0]+"자료는" + data[1]);
	%>
</body>
</html>
```
<br>
결과 URL:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/Forward1.JPG" height="100" width="700" /></div>
<br>
결과:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/Forward2.JPG" height="150" width="700" /></div>
<br>

###  Redirect
Redirect방식으로 요청을 처리하는 방식이다.  
1) Redirect.java
 - Redirect라는 요청이 들어왔을때 요청하는 곳이다. 
 - name변수에 HTML에서 data의 이름을 가진 변수의 값을 저장한다.
 - Forward.jsp에 name을 전달한다.

```java
package pack;

import java.io.IOException;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Servlet implementation class Redirect
 */
@WebServlet("/Redirect")
public class Redirect extends HttpServlet {
	protected void service(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		request.setCharacterEncoding("utf-8");
		String name = request.getParameter("data");
		
		//Redirect - Client to Server
		response.sendRedirect("Redirect.jsp?name=" + name);
		
	}

}

```

2) Redirect.jsp
 - 받은 자료를 출력하는 곳 이다.

```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
	Called File by Servlet
	<br>
	<%
		//redirect Method
		String name = request.getParameter("name");
		out.println("방식은 Redirect 자료는 " + name);
		
	%>
</body>
</html>
```
<br>
결과 URL:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/Redirect1.JPG" height="100" width="700" /></div>
<br>
결과:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/Redirect2.JPG" height="150" width="700" /></div>

<br>

<hr>
참조:<a href="https://github.com/wjddyd66/Web/tree/master/Forward_Redirect">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
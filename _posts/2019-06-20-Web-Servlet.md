---
layout: post
title:  "Web-Servlet"
date:   2019-06-20 08:30:00 +0700
categories: [Web]
---

###  Servlet
웹기반 요청에 대한 동적인 처리가 가능한 하나의 클래스 이다.  
Server Side에서 돌아가는 Java Program이다.  
HTML Form Element안에 정보를 담고 Input Element에 의해 정보 전송  
Form Element
 - method: 원하는 동작 설정: get or post
 - action: Servlet에게 전송되는 논리적인 URL


Input Element
 - type="submit": Form 안의 내용을 해당 URL에 맞는 Servlet 요청으로 들어가게 함

###  Html
get과 post방식으로 hi.do라는 URL로 통해 보내게 된다.   


```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Servlet</title>
</head>
<body>
*서블릿 연습*<p/>
<form action="hi.do" method="get">
	<input type="submit" value="전송-get">
</form>
<br>
<form action="hi.do" method="post">
	<input type="submit" value="전송-post">
</form>
<br>
</body>
</html>
```
###  Servlet - Java
hi.do라는 URL요청이 왔을 경우 처리하는 곳 이다.  
1) init()
 - 최초 접속자에 의해 1회 수행되는 초기화 작업이다. 
 - Servlet 객체를 초기화하는 역할이다(Servlet 객체를 메모리에 할당한다.)
 - 1초뒤 sendKeyword() 실행
 - 1초의 텀을 둔 이유는 한글에서 한 글자를 적을때까지 기달리게 하기 위해서 이다.

2) Service(request,response)
 - 응답에 대한 모든 내용이 구현되는 곳 입니다.
 - doget()
  - Service중 doget Method를 Override한 곳이다.
  -  get방식이 들어왔을때 수행되는 함수이다. 
  - 서블릿 strart - doGet라는 HTML 문서로 Return 되어 보여지게 된다.
 - dopost()
  - Service중 dopost Method를 Override한 곳이다.
  -  post방식이 들어왔을때 수행되는 함수이다. 
  - 서블릿 strart - doPost라는 HTML 문서로 Return 되어 보여지게 된다.

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>do</td><td>post</td>
	</tr>
	<tr>
		<td>스트링 형태로 전송</td><td>인코딩 형태로 전송</td>
	</tr>
		<tr>
		<td>URL에 정보 보임</td><td>URL에 정보 안보임</td>
	</tr>
		<tr>
		<td>공유가 쉬움</td><td>공유가 어려움(정보가 인코딩 되어 안보이기 때문에 URL 링크다고 이동시 정보가 없는 결과가 나오게 된다.)</td>
	</tr>
	</tbody>
</table>
<br>

3) destroy()
 - 서비스 종료시 1회 수행되는 함수이다.
 - Servlet 객체를 메모리에서 제거한다.
<div><img src="https://gmlwjd9405.github.io/images/web/servlet-program.png" height="300" width="700" /></div>
<br>
출처: <a href="https://gmlwjd9405.github.io/2018/10/28/servlet.html">heejeong Kwon블로그</a><br><br>


Servlet 코드
```java
package pack;

import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;


@WebServlet("/hi.do")
public class TestServlet extends HttpServlet {
	
	public void init(ServletConfig config) throws ServletException {
		// 최초접속자에 의해 1최 수행 - 초기화작업
		System.out.println("초기화 작업");
		
	}

	
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		response.setContentType("text/html;charset=utf-8");
		PrintWriter out = response.getWriter();
		out.println("<html><body>");
		out.println("<h1>서블릿 start -doGet</h1>");
		out.println("</body></html>");
		out.close();
	}

	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		response.setContentType("text/html;charset=utf-8");
		PrintWriter out = response.getWriter();
		out.println("<html><body>");
		out.println("<h1>서블릿 start -doPost</h1>");
		out.println("</body></html>");
		out.close();
	}

	public void destroy() {
		//서비스 종료 시 1회 수행 - 마무리 담당
		System.out.println("destory");
	}
}
```

###  결과
아래 동영상은 Get, Post방식에 따른 출력 결과이다.  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/179d1b78de5a44bf9df0cf0c9eb74ac8" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
아래 사진은 생성자에 의해 한번만 수행되는 것을 나타낸 결과이다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js44.JPG" height="150" width="600" /></div>
<br>

<hr>
내용참조: <a href="https://mangkyu.tistory.com/14">망나니개발자 블로그</a><br>
내용참조: <a href="https://gmlwjd9405.github.io/2018/10/28/servlet.html">heejeong Kwon블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Web/tree/master/Servlet">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
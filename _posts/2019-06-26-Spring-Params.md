---
layout: post
title:  "Spring-요청방식"
date:   2019-06-26 12:10:00 +0700
categories: [Spring]
---

###  Spring-요청방식
Spring 에서 Controller에게 요청하는 방식은 2가지이다.  
1. Get
2. Post

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Get</td><td>Post</td>
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
값을 요청받거나 주었을때 한글이 깨지는 현상은 아래 코드를 Web.xml에 추가함으로써 해결할 수 있다.  
```xml
 <filter>  
        <filter-name>encodingFilter</filter-name>  
        <filter-class>org.springframework.web.filter.CharacterEncodingFilter</filter-class>  
        <init-param>   
            <param-name>encoding</param-name>   
            <param-value>UTF-8</param-value>  
        </init-param> 
    </filter> 
    <filter-mapping>  
        <filter-name>encodingFilter</filter-name>
        <url-pattern>/*</url-pattern>
```

###  index.jsp,showMessage.jsp
index.jsp  
요청을 보내기 위한 첫 페이지이다.  
관리자, 일반회원, 파라미터 없음은 URL에서 Type의 값을 비교하여 값이 달라지는 것을 보여주기 위한 요청이다.  
전송 ~ 전송4는 각각 Get 과 Post방식으로 값을 보냈을 경우 어떻게 받아서 처리하는 지에 대한 요청이다.  
showMessage.jsp  
index.jsp -> Controller를 거친뒤의 결과를 보여주는 곳 이다.  

```jsp
<!-- index.jsp -->
<!DOCTYPE html>

<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
    
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<%@ taglib prefix="spring" uri="http://www.springframework.org/tags"%>

<html>
	<head>
		<meta charset="utf-8">
		<title>Welcome</title>
	</head> 
	<body>
	요청 파라미터 연습*<br>
	<a href="kic/login?type=admin">관리자</a>
	<a href="kic/login?type=user">일반회원</a>
	<a href="kic/login">파라미터없음</a>
	<br>
	<form action="kic/login" method="post">
		data: <input type="text" name="type" value="Hwang">
		<input type="submit" value="전송">
	</form><br>
	<form action="kic/hello" method="post">
		data: <input type="text" name="type" value="Jeong">
		<input type="submit" value="전송2">
	</form>
	
	<form action="hello/get/world/Java" method="get">
		신곡: <input type="text" name="title" value="봄이 와요">
		<input type="submit" value="전송3">
	</form>
	
	<form action="hello/get/world/Spring" method="get">
		신곡: <input type="text" name="title" value="여름이 와요">
		<input type="submit" value="전송4">
	</form>
	
	</body>
</html>

<!-- showMessage.jsp -->
<!DOCTYPE html>

<%@ page language="java" contentType="text/html; charset=UTF-8" pageEncoding="UTF-8"%>
    
<html>
	<head>
		<meta charset="utf-8">
		<title>Welcome</title>
	</head> 
	<body>
		<h2>${message}</h2>
	</body>
</html>

```
<br>
###  loginController.java
index.jsp에서 받은 요청을 처리하는 곳 이다.  
1. admin(),user(),etc(): kic/login뒤에 붙은 변수 type으로 써 구별한다.  
2. post(): Post방식으로 들어오는 것을 처리한다.  
3. get(): Get방식으로 들어오는 것을 처리한다.  


@Controller: Controller라는 것을 명시  
@RequestMapping: index.jsp의 요청을 받기 위하여 사용(value: 실제 요청 내용, params: 받은 변수 내용)  
@RequestParam: 요청에 대해 매칭되는 request parameter값이 자동으로 들어감  
@PathVariable: HTTP 요청에 대해 매칭되는 request parameter값이 자동으로 들어감   
ModelAndView: ViewResolver에 전달할 View 이름이다. Spring에서는 WEB-INF/view에 jsp형태로 자동으로 찾아가게 된다. ModelandView admin()을 예로 들면  

ModelAndView view = new ModelAndView("showMessage"); // showMessage.jsp로 가라  
view.addObject("message", "관리자");//showMessage로 갈때 message라는 변수에 "관리자"라는 값을 담아서 가라  
return view; // showMessage.jsp로 이동  

```java
package pack;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.ModelAndView;

@Controller
public class loginController {

	@RequestMapping(value = "kic/login", params = "type=admin")
	public ModelAndView admin() {
		ModelAndView view = new ModelAndView("showMessage");
		view.addObject("message", "관리자");
		return view;
	}

	@RequestMapping(value = "kic/login", params = "type=user")
	public ModelAndView user() {
		ModelAndView view = new ModelAndView("showMessage");
		view.addObject("message", "유저");
		return view;
	}

	@RequestMapping(value = "kic/login", params = "!type")
	public ModelAndView etc() {
		ModelAndView view = new ModelAndView("showMessage");
		view.addObject("message", "기타");
		return view;
	}

	@RequestMapping(value = "kic/{url}")
	public ModelAndView post(@RequestParam("type") String type, @PathVariable String url) {
		ModelAndView view = new ModelAndView("showMessage");
		view.addObject("message", type + url);
		return view;
	}
	
	@RequestMapping(value = "hello/{para1}/world/{para2}")
	public ModelAndView get(@RequestParam("title") String title, @PathVariable("para1") String para,@PathVariable String para2) {
		ModelAndView view = new ModelAndView("showMessage");
		view.addObject("message", title + para+para2);
		return view;
	}
}

```
<br>

실행결과  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/603c4983ffb64471ad370ae9aa970a56" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

<hr>
참조:<a href="https://github.com/wjddyd66/Spring/tree/master/Params">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
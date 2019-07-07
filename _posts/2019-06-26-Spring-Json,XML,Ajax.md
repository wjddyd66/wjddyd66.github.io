---
layout: post
title:  "Spring-XML,JSON,Ajax"
date:   2019-06-26 12:20:00 +0700
categories: [Spring]
---

###  Spring-XML,JSON,Ajax
JSON(JavaScript Object Notation): 좀 더 쉽게 데이터를 교환하고 저장하기 위하여 만들어진 텍스트 기반의 데이터 교환 표준  
XML(EXtensible Markup Language): HTML과 매우 비슷한 문자 기반의 마크업 언어 이다. HTML과 달리 데이터를 보여주는 목적이 아닌, 데이터를 저장하고 전달할 목적으로 만들어졌다. Tag는 미리 지정되어있지 않고 사용자가 직접 정의할 수 있다.

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>공통점</td><td>차이점</td>
	</tr>
	<tr>
		<td>데이터를 저장하고 전달하기 위해 고안</td><td>JSON은 종료 태그가 없다</td>
	</tr>
		<tr>
		<td>가독성이 뛰어남</td><td>JSON이 XML보다 짧다</td>
	</tr>
		<tr>
		<td>계층적인 데이터 구조</td><td>JSON이 XML데이터보다 더 빨리 읽고 쓸 수 있다.</td>
	</tr>
			<tr>
		<td>프로그래밍 언어에 의해 파싱 가능</td><td>JSON은 배열을 사용할 수 있다.</td>
	</tr>
			<tr>
		<td>HTTP Request객체를 이용하여 서버로부터 데이터를 전송 받을 수 있다.</td><td></td>
	</tr>
	</tbody>
</table>
<br>
<span style ="color: red">**JSON은 문자열이므로 XML보다 빠르고 장점이 많지만, 사용자가 직접 데이터의 무결성을 검증하여야 한다.따라서 데이터 검증이나 스키마를 사용하여 무결성을 검증할 수 있는 XML도 많이 사용이 된다.**</span>
<br>
<br>
Ajax( Asynchronous Javascript And Xml)  
<a href="https://wjddyd66.github.io/web/2019/06/20/JavaScript-DOM,JQuery,Ajax.html">AJAX 참고</a>  
Ajax통신에서 사용하는 Parameter  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>주요 속성</td><td>설명</td>
	</tr>
	<tr>
		<td>data</td><td>서버에 전송할 데이터, key/value 형식의 객체</td>
	</tr>
	<tr>
		<td>dataType</td><td>서버가 리턴하는 데이터 타입(xml,json,script,html)</td>
	</tr>
	<tr>
		<td>type</td><td>서버가 전송하는 데이터의 타입(Post,GET)</td>
	</tr>
	<tr>
		<td>url</td><td>데이터를 전송할 URL</td>
	</tr>
	<tr>
		<td>success</td><td>Ajax통신에 성공했을때 호출될 이벤트 핸들러</td>
	</tr>
	</tbody>
</table>
<br>


###  XML - myform.jsp
myform.jsp는 XML로 Return하는 값을 요청하기 위하여 값을 넣고 요청하는 곳이다.  
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
	<h1>자료입력</h1>
	<h3>Xml로 처리</h3>
	<form action="member_xml" method="post">
		Name: <input type="text" name="name"><br>
		Age: <input type="text" name="age"><br>
		<input type="submit" value="OK">
	</form>
	<br>
</body>
</html>
```
<br>
###  XML - pack.Model
xmlMessage.java: 요청한 정보를 전달하기 위해 만들어진 객체이다.(DTO)  
xmlMessageList: XML로 Return하기 위해 정의한 곳 이다.  

```java
//xmlMessage.java
package pack.Model;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;

@XmlAccessorType(XmlAccessType.FIELD)
public class xmlMessage {//dto
	private String name, age;

	public xmlMessage(String name, String age) {
		this.name=name;
		this.age=age;
	}
	
	public String getName() {
		return name;
	}
	
	public String getAge() {
		return age;
	}
}

//XmlMessageList
package pack.Model;

import java.util.List;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

@XmlAccessorType(XmlAccessType.FIELD)
@XmlRootElement(name="msg_list") //RootElement의 이름
public class XmlMessageList {
	@XmlElement(name="msg")//Element 의 이름
	private List<xmlMessage> message;
	
	public XmlMessageList() {
		
	}
	
	public XmlMessageList(List<xmlMessage> message) {
		this.message=message;
	}
	
	public List<xmlMessage> getMessage(){
		return message;
	}
	
}

```
###  XML - pack.Controller
요청이 들어왔을때 실질적으로 처리하는 부분이다.  
위에서 선언한 XmlMessageList에 정보를 담아서 XML로서 Return처리를 하는 곳 이다.  

```java
package pack.Controller;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.Arrays;
import java.util.List;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import pack.Model.XmlMessageList;
import pack.Model.xmlMessage;

@Controller
public class xmlController {
	
	@RequestMapping(value="member_xml", method=RequestMethod.GET)
	public String formBack() {
		return "myform";
	}
	
	@RequestMapping(value="member_xml", method=RequestMethod.POST)
	@ResponseBody
	public XmlMessageList submit(@RequestParam("name") String name, @RequestParam("age") String age) {
		return getXml(name, age);
	}
	
	public XmlMessageList getXml(String name, String age) {
		List<xmlMessage> messages=Arrays.asList(
				new xmlMessage(name, age), 
				new xmlMessage("Oscar", "33"),
				new xmlMessage("Tom", "44")
				);
		return new XmlMessageList(messages);
	}
}
```
<br>
실행결과:  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/XML.JPG" height="200" width="700" /></div>
<br>

###  JSON,Ajax - index.jsp
Ajax를 활용하여 결과를 얻을때 Return Type을 Json으로 받은뒤 처리하는 예제이다.  
btnOk1은 Ajax 통신을 통하여 한개의 자료를 그냥 받아서 처리하는 과정 이다.  
btnOk2를 Ajax 통신을 통하여 다량의 자료를 Json 타입으로 받아서 처리하는 과정이다.  

```code
<!DOCTYPE html>

<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>

<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core"%>
<%@ taglib prefix="spring" uri="http://www.springframework.org/tags"%>

<html>
<head>
<meta charset="utf-8">
<title>Welcome</title>
<script
	src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
<script type="text/javascript">
	$(document).ready(function() {
		$("#btnOk1").click(function() {
			$.ajax({
				url : "list",
				type : "GET",
				data : {
					"name" : "tom"
				},
				success : function(Data) {
				var str="";
				str+=Data.name+"<br>";
				str+=Data.skills[0]+" "+Data.skills[1];
				$("#showData").html(str);
				}
			});
		});

		$("#btnOk2").click(function() {
			$.ajax({
				url : "list2",
				type : "GET",
				dataType:"json",
				success : function(Data) {
				alert(Data);
				var str="<table>";
				var list= Data.datas;
				$(list).each(function(index,obj){
					str+="<tr>";
					str+="<td>"+obj["name"]+"</td>";
					str+="<td>"+obj["age"]+"</td>";
					str+="</tr>";
				});
				str+="</table>";
				console.log(str);
				$("#showData").html(str);
				}
			});
		});
	})
</script>
</head>
<body>
	<a href="list?name=james">json처리: 단일자료</a>
	<br>
	<input type="button" value="한 개 자료" id="btnOk1">
	<br>
	<input type="button" value="복수 자료" id="btnOk2">
	<br>
	<div id="showData"></div>
</body>
</html>
```
<br>

###  JSON,Ajax - pack.Controller
jsonController: 하나의 자료를 처리하는 Controller이다.  
jsonController2: 다수의 자료를 처리하는 Controller이다.  
둘다 반환하는 형식은 비슷하지만 Ajax 통신을 설정할때 dataType을 어떻게 설정하냐에 따라서 Return Type이 결정나는 것이다.  


```java
//JsonController
package pack.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import pack.model.MyModel;

@Controller
@RequestMapping("list")
public class JsonController {

	@Autowired
	private MyModel myModel;
	
	@RequestMapping(method=RequestMethod.GET)
	@ResponseBody
	public MyModel getJson(@RequestParam("name") String name) {
		myModel.setName(name);
		myModel.setSkills(new String[] {"자바 전문 개발자","DB운영 숙련자"});
		return myModel;
	}
}

//JsonController2
package pack.controller;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import pack.model.MyModel;

@Controller
@RequestMapping("list2")
public class JsonController2 {

	@Autowired
	private MyModel myModel;
	
	@RequestMapping(method=RequestMethod.GET)
	@ResponseBody
	public Map getJson() {
		List<Map<String, String>> datalist = new ArrayList<Map<String,String>>();
		Map<String,String> data = new HashMap<String, String>();
		data.put("name","홍길동");
		data.put("age","20");
		datalist.add(data);
		
		data = new HashMap<String, String>();
		data.put("name","한국인");
		data.put("age","25");
		datalist.add(data);
		
		data = new HashMap<String, String>();
		data.put("name","신기해");
		data.put("age","35");
		datalist.add(data);
		
		Map<String,Object> data2 = new HashMap<String,Object>();
		data2.put("datas",datalist);
		return data2;
	}
}

```
<br>

###  JSON,Ajax - pack.model
자료를 저장하고 옮기는 저장소를 하는 형태를 정하는 곳이다.(DTO)  

```java
package pack.model;
import org.springframework.stereotype.Component;

@Component
public class MyModel {
	private String name;
	private String skills[];
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String[] getSkills() {
		return skills;
	}
	public void setSkills(String[] skills) {
		this.skills = skills;
	}
	
}
```
<br>


실행결과  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/b9e1f1818147434dbd7845e7e9bcb2f6" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

<hr>
내용참조:<a href="http://tcpschool.com/json/json_intro_xml">TCP School</a><br>
참조:<a href="https://github.com/wjddyd66/Spring/tree/master/Json%2CAjax">원본코드(Json)</a><br>
참조:<a href="https://github.com/wjddyd66/Spring/tree/master/XML">원본코드(XML)</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
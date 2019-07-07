---
layout: post
title:  "Spring-기본 개념"
date:   2019-06-26 06:30:00 +0700
categories: [Spring]
---

###  Spring
스프링이란 자바 엔터프라이즈 개발을 편하게 해주는 오픈소스 경량급 애플리케이션 프레임 워크이다. 즉, 애플리케이션 개발의 전 과정을 빠르고 편리하며 효율적으로 진행하는데 일차적인 목표를 두는 프레임 워크이다.  
Spring은 단순한 개발툴과 기본적인 개발환경으로도 엔터프라이즈 개발에서 필요로 하는 주요한 기능을 갖춘 애플리케이션을 개발하기 충분하다. 즉, 만들어진 코드가 지원하는 기술수준이 비슷하더라도 더 빠르고 간편하게 개발이 가능하게 해주는 프레임워크이다.  

###  POJO(Plain Old Java Object)
POJO에대해 알아보기 전에 탄생하게 된 배경을 알아보자.  
POJO를 사용하지 않던 시절에는 개발자가 비지니스 로직 외에 트랜잭션, 멀티스레드, 보안 등 여러가지 기능을 직접 jdk를 사용하여 구현하였다.  
이러한 것을 보안하기 위하여 sum사에서 EJB를 만들었다. EJB를 활용함으로써 애플리케이션 개발자는 로우레벨의 기술에는 관심이 가질 필요가 없었다. 이러한 장점에도 불고하고 비즈니스 오브젝트들을 객체지향적인 특징과 장점을 포기해야하는 큰 단점이 생기게 되었다.  
EJB의 객체지향적인 특징과 장점을 못살리는 단점을 극복하기 위한 것이 POJO이다.  
POJO를 사용하게 되면 복잡한 로우레벨의 API를 이용해야 하는 코드를 작성해야 하지만 다음과 같은 장점이 생기게 된다.  
1. 코드의 간결함(비즈니스 로직과 특정 환경/low 레벨 종속적인 코드로 분리)
2. 자동화 테스트 유리
3. 객체지향적 설계의 자유로운 사용

이러한 POJO에서 제공하는 객체지향적인 장점과 EJB의 제공하는 엔터프라이즈 서비스와 기술을 그대로 사용할 수 있도록 도와주는 프레임워크가 POJO 프레임 워크이다.  
Sping은 이러한 POJO프레임 워크중에 하나이다.  
(Spring에서는 POJO를 Beans라고 부른다.)
###  Bean Mapping 방법
bean을 사용하기 위하여 Mapping하는 방법은 3가지가 있다.
1. URL
2. Pattern
3. Annotation

```xml
    <!-- Mapping 1 : url-->
	<!-- 
	<bean class="org.springframework.web.servlet.handler.BeanNameUrlHandlerMapping"/>
	
	<bean id="/hello.do" name="/hi.do, /abc/world.do" class="pack.controller.HelloController">
		<property name="helloModel" ref="helloModel"/>
	</bean>
	<bean id="helloModel" class="pack.model.HelloModel"/>
	
	<bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
		<property name="prefix" value="/views/"/>
		<property name="suffix" value=".jsp"/>
	</bean>
	 -->
	 
	 <!-- Mapping 2 : pattern (?, *) -->
	 <!-- 
	 <bean class="org.springframework.web.servlet.handler.SimpleUrlHandlerMapping">
	 	<property name="alwaysUseFullPath" value="true"/>
	 	<property name="mappings">
	 		<props>
	 			<prop key="*.do">hi</prop>
	 			<prop key="/**/?????.do">hi</prop>
	 		</props>
	 	</property>
	 </bean>
	 
	 <bean name="hi" class="pack.controller.HelloController">
	 	<property name="helloModel" ref="helloModel"/>
	 </bean>
	 
	 <bean id="helloModel" class="pack.model.HelloModel"/>
	 
	 <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
	 	<property name="prefix" value="/views/"/>
		<property name="suffix" value=".jsp"/>
	 </bean>
	  -->
	 
	 <!-- Mapping 3 : Using Annotation -->
	 <context:component-scan base-package="pack.controller"/>
	 <context:component-scan base-package="pack.model"/>
	 
	 <bean class="org.springframework.web.servlet.view.InternalResourceViewResolver">
	 	<property name="prefix" value="/views/"/>
		<property name="suffix" value=".jsp"/>
	 </bean>
```
###  Bean Scope
스프링은 기본적으로 bean을 하나의 Singletone객체로 설정한다.  
bean의 Scope를 사용하여 다른 범위로 사용할 수 있다.  
<a href="https://wjddyd66.github.io/others/2019/06/14/SingleTon.html">SingleTon 자세한 내용</a>
<br>
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>Scope</td><td>설명</td>
	</tr>
	<tr>
		<td>SingleTon</td><td>하나의 Bean정의에 대해서 단 하나의 객체만 존재</td>
	</tr>
		<tr>
		<td>Prototype</td><td>하나의 Bean정의에 대해서 다수의 객체 존재 가능</td>
	</tr>
		<tr>
		<td>Request</td><td>하나의 Bean정의에 대해서 하나의 HTTP request의 생명주기 안에 단 하나의 객체만 존재</td>
	</tr>
			<tr>
		<td>Session</td><td>하나의 Bean정의에 대해서 하나의 HTTP Session의 생명주기 안에 단 하나의 객체만 존재</td>
	</tr>
	<tr>
		<td>Global Session</td><td>하나의 Bean정의에 대해서 하나의 Global HTTP Session의 생명주기 안에 단 하나의 객체만 존재</td>
	</tr>
	</tbody>
</table>
<br>
###  MVC패턴
Spring은 POJO프레임 워크이다. 이러한 Spring은 클래스를 관리하는 역할을 가지게 되며, 의존성이 약한 프로그램이다.  
객체가 서로 강한 결합으로 묶여있으면 유지보수가 힘들어지게 된다.  
Spring은 인터페이스 기반의 설계를 하고, 어떤 클래스를 사용시 구상 클래스가 아닌 인터페이스를 통해 사용하려는 클래스의 메서드를 호출한다.  
따라서 객체가 서로 약한 결합으로 연결되므로 유지보수에 용이하다.  
아래 그림은 Spring MVC 구조를 이해하기 위한 그림이다.  

<div><img src="https://mblogthumb-phinf.pstatic.net/20160512_289/lakeni_1463025116804Q2uGQ_PNG/%B1%D7%B8%B21.png?type=w800" height="400" width="600" /></div>
그림참조:<a href="https://m.blog.naver.com/PostView.nhn?blogId=lakeni&logNo=220708587953&proxyReferer=https%3A%2F%2Fwww.google.com%2F">도란비 블로그</a>
<br>
위와 같은 그림의 흐름을 약결합으로 인해 분리하고 유지보수가 쉽게 하기 위해서 MVC라는 패턴을 사용하는 것이다.   
이러한 분리된 M,V,C를 DispatcherServlet을 활용하여 중앙 통제식 구조로 사용하게 된다.  


1. Model: 애플리케이션의 정보, 데이터
2. View: 사용자에게 화면을 보여주는 인터페이스
3. Controller: 비지니스 로직과 모델의 상호동작의 조정 역할을 한다.


<br>
###  Maven
Maven이란 사용할 라이브러리뿐만 아니라 해당 라이브러리가 작동하는 필요한 다른 라이브러리들까지 관리하여 네트워크를 통해서 자동으로 다운받아주는 것 이다.  
이로 인하여 프로젝트 전체적인 라이프 사이클을 관리하는 도구로서 사용하게 된다.  
<table class="table">
	<tbody>
	<tr>
		<td>장점</td><td>단점</td>
	</tr>
	<tr>
		<td>컴파일과 빌드를 동시에 수행 가능</td><td>기본적으로 지원하지 않는 빌드 과정을 추가 하여야 한다.</td>
	</tr>
		<tr>
		<td>pom.xml파일을 통해 관리하므로 개발, 유지보수 측면에서 오픈소스 라이브러리, 프로젝트 등 관리가 용이하다.</td><td>플러그인이 설정이 약간만 달라도 해당 설정을 분리해서 중복 기술할 해야 한다.</td>
	</tr>
		<tr>
		<td>IDE에 종속된 부분들을 제거 가능</td><td></td>
	</tr>
	</tbody>
</table>
<br>

<br>
<hr>
내용참조:<a href="https://12bme.tistory.com/157">12bme 블로그</a><br>
내용참조:<a href="https://limmmee.tistory.com/8">dongdong_ 블로그</a><br>
내용참조:<a href="https://gmlwjd9405.github.io/2018/11/10/spring-beans.html">gmlwjd9405 블로그</a><br>
내용참조:<a href="https://jerryjerryjerry.tistory.com/63">쩨리쪠리 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
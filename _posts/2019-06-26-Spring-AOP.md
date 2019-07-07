---
layout: post
title:  "Spring-Annotaion,AOP"
date:   2019-06-26 12:00:00 +0700
categories: [Spring]
---

###  Annotation
AOP를 들어가기 앞서 사전지식이 필요한 개념이다.  
Annotaion으로 인하여 데이터의 유효성 검사 등을 쉽게 할 수 있고, 이에 관련된 코드가 깔끔해지게 된다.  
Annotaion으로 인하여 AOP를 편리하게 구성할 수 있게 하며 실제 데이터가 아닌 Data를 위한 데이터, 즉, Meta Data이다.  

###  Java의 기본적인 Annotation
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Annotaion</td><td>설명</td>
	</tr>
	<tr>
		<td>@Override</td><td>
		<ul>
			<li>선언한 Method가 Override되었다는 것을 나타냄</li>
			<li>부모 클래스에서 해당 메서드를 찾을 수 없다면 Error 발생</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Deprecated</td><td>
		<ul>
			<li>해당 메서드가 더 이상 사용되지 않음</li>
			<li>만약 사용할 경우 컴파일 경고를 발생</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Suppress Warnings</td><td>
		<ul>
			<li>선언한 곳의 컴파일 경고를 무시하도록 함</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@SafeVarargs</td><td>
		<ul>
			<li>제너릭 같은 가변인자의 매개변수를 사용할 때의 경고를 무시한다.</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@FunctionalInterface</td><td>
		<ul>
			<li>함수형 인터페이스를 지정</li>
			<li>메서드가 존재하지 않거나, 1개 이상의 메서드가 존재할 경우 컴파일 오류 발생</li>
		</ul>
		</td>
	</tr>
	</tbody>
</table>
<br>

###  Meta Annotation
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Annotaion</td><td>설명</td>
	</tr>
	<tr>
		<td>@Retention</td><td>
		<ul>
			<li>자바 컴파일러가 어노테이션을 다루는 방법을 기술, 특정 시점까지 영향을 미치는지를 결정</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Target</td><td>
		<ul>
			<li>어노테이션이 적용할 위치를 선택</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Documented</td><td>
		<ul>
			<li>해당 어노테이션을 Javadoc에 포함</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Inherited</td><td>
		<ul>
			<li>어노테이션의 상속을 가능하게 한다.</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Repeatable</td><td>
		<ul>
			<li>연속적으로 어노테이션을 선언</li>

		</ul>
		</td>
	</tr>
	</tbody>
</table>
<br>

###  Spring Annotation
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Annotaion</td><td>설명</td>
	</tr>
	<tr>
		<td>@Component</td><td>
		<ul>
			<li>component-scan을 선언에 의해 특정 패키지 안의 클래스들을 스캔하고 @Component Annotaion이 있는 클래스에 대하여 bean인스턴스를 생성</li>
			<li>@Controller: Presentation Layer에서 Controller를 명시하기 위해 사용</li>
			<li>@Service: Business Layer에서 Service를 명시하기 위해 사용</li>
			<li>@Repository: Presentation Layer에서 DAO를 명시하기 위해 사용</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@RequestMapping</td><td>
		<ul>
			<li>Class Level Mapping: 특정 요청에 대한 처리를 해당 클래스에서 한다는 것</li>
			<li>Handler Level Mapping: 특정 요청이 Post, Get인지를 구분하고 해당하는 요청에 맞을때 수행</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Required</td><td>
		<ul>
			<li>setter Method에 사용. property를 채워야 한다.</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Autowired</td><td>
		<ul>
			<li>Type에 따라 알아서 Bean을 주입</li>
			<li>Type을 확인하후 못찾으면 Name에 따라 주입</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Qualifier</td><td>
		<ul>
			<li>같은 타입의 빈이 두 개 이상이 존재하는 경우에 Spring이 어떤 빈을 주입해야 할지 알 수 없어서 명확히 지시</li>

		</ul>
		</td>
	</tr>
	<tr>
		<td>@RequestParam</td><td>
		<ul>
			<li>GET요청에 대해 매칭되는 request parameter값이 자동으로 들어감</li>
			<li>URL 뒤의 문자열이 실제 전달되는 이름 값이다.</li>
	
		</ul>
		</td>
	</tr>
	<tr>
		<td>@Path Variable</td><td>
		<ul>
			<li>HTTP 요청에 대해 매칭되는 request parameter값이 자동으로 들어감</li>
			<li>URL 에서 각 구분자에 들어오는 값을 처리할 때 사용</li>
	
		</ul>
		</td>
	</tr>
	<tr>
		<td>@RequestBody</td><td>
		<ul>
			<li>POST요청에 대해 매칭되는 request message값이 자동으로 들어감</li>
		</ul>
		</td>
	</tr>
	<tr>
		<td>@ModelAttribute</td><td>
		<ul>
			<li>Form 값이 자동으로 Mapping 된다.</li>
		</ul>
		</td>
	</tr>
	</tbody>
</table>
<br>

###  AOP(Aspect Oriented Programming)
AOP란 관점지향 프로그래밍 이다.  
공통적인 기능을 모든 모듈에 적용하기 위한 방법으로 상속을 이용한다.  
JAVA에서는 다중상속이 불가능하기 때문에 AOP로서 한계를 극복한다.  
<span style ="color: red">**이러한 AOP의 핵심기능은 공통 기능을 분리시키고 공통 기능을 필요로 하는 기능들에서 사용하는 방식이다.**</span><br>

###  AOP Annotaion

<table class="table">
	<tbody>
	<tr>
		<td>구성 요소</td><td>설명</td>
	</tr>
	<tr>
		<td>JoinPoint</td><td>관심 모듈의 기능이 삽입되어 동잘 할 수 있는 실행 가능한 특정 위치</td>
	</tr>
		<tr>
		<td>Pointcut</td><td>어떤 클래스의 어느 조인포인트를 사용할 것인지를 선택 가능</td>
	</tr>
		<tr>
		<td>Weaving</td><td>포인트컷에 의해서 결정된 조인포인트에 지정된 어드바이스를 삽입하는 과정</td>
	</tr>
			<tr>
		<td>Aspect</td><td>Pointcut에서 Advice를 할 것인지</td>
	</tr>
	</tbody>
</table>
<br>
<div><img src="https://t1.daumcdn.net/cfile/tistory/185DF4334FA8A1FD01" height="400" width="600" /></div>
그림참조:<a href="https://isstory83.tistory.com/90">I's Stroy 블로그</a>
<br>

###  init.xml
AOP를 사용하기 위한 환경 설정 이다.  
<
aop:aspectj-autoproxy 
/>: Java AOP를 사용하기 위하여 선언  
<
context:component-scan base-package="pack" 
/>: 일일이 Bean을 등록하지 않고 사용하기 위하여 선언  

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
	xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xmlns:c="http://www.springframework.org/schema/c"
	xmlns:context="http://www.springframework.org/schema/context"
	xmlns:p="http://www.springframework.org/schema/p"
	xmlns:aop="http://www.springframework.org/schema/aop"
	xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans-4.3.xsd
		http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context-4.3.xsd
		http://www.springframework.org/schema/aop http://www.springframework.org/schema/aop/spring-aop-4.3.xsd">

	<context:component-scan base-package="pack" />
	
	<aop:aspectj-autoproxy />
</beans>
```
<br>
###  pack.model
DB와 연관되는 작업을 하는 곳이다.  
ArticleInter: Interface를 사용하여 공동 작업시 충돌 방지  
ArticleDAO: 실제 DB와 연결하여 작업(현재는 DB와 연결되어 있지 않으므로 간단한 출력형식을 사용하였다.)

```java
//ArticleInter
package pack.model;

public interface ArticleInter {
	void selectAll();
}

//ArticleDao
package pack.model;

import org.springframework.stereotype.Repository;

@Repository("articleDao")
public class ArticleDao implements ArticleInter{
	
	@Override
	public void selectAll() {
		System.out.println("직원 테이블 전체자료 조사");
		
	}

}
```

###  pack.BL
실제 Logic이 구현되는 곳이다.  
LogicInter: Interface를 사용하여 공동 작업시 충돌 방지  
LogicImpl: 실제 Logic이 실행되는 곳이다.  
 - Qualifier를 통하여 ArticleDao를 객체화 하여 사용
 - articleInter.selectAll()를 통하여 DB와 연동되어 작업
Main: 실제 수행을 위하여 실행을 하는 곳

```java
//LogicInter
package pack.BL;

public interface LogicInter {
	void selectdataProcess();
}

//LogicImpl
package pack.BL;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import pack.model.ArticleInter;

@Service
public class LogicImpl implements LogicInter {
	@Autowired
	@Qualifier("articleDao")
	private ArticleInter articleInter;

	
	@Override
	public void selectdataProcess() {
		System.out.println("selectdataProcess 작업하는 중...");
		articleInter.selectAll();
	}
	
}

//Main
package pack.BL;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class Main {

	public static void main(String[] args) {
		ApplicationContext context = new ClassPathXmlApplicationContext("init.xml");
		LogicInter inter = context.getBean("logicImpl",LogicInter.class);
		inter.selectdataProcess();

	}

}
```
###  pack.Aspect
AOP를 사용하는 곳 이다.  
ASPECT는 다음과 같은 형태로 정리될 수 있다.  
```code
// AspectJ의 Pointcut 표현식 정리

execution([접근자제어패턴], 리턴타입패턴 [패키지패턴]메서드이름패턴(파라메터패턴)) [ ] 안의 패턴은 생략 가능

execution(public void set*(..))
public에 리턴값이 없으며, 패키지명은 없고, 메서드는 set으로 시작하며 인자값은 0개 이상인 메서드 호출

execution(* com.people.*.*())
리턴타입에 상관없이 com.people패키지의 인자값이 없는 모든 메서드 호출

execution(* com.people..*.*(..))
리턴타입에 상관없이 com.people 패키지 및 하위 패키지에 있는, 인자값이 0개 이상인 메서드 호출

execution(Integer com.people.WriteService.write(..))
리턴 타입이 Integer인 WriteServlce의 인자값이 0개 이상인 write() 호출

execution(* get*(*))
메서드 이름이 get으로 시작하는 인자값이 1개인 메서드 호출

execution(* get*(*,*))
메서드 이름이 get으로 시작하는 인자값이 2개인 메서드 호출

execution(* get*(Integer, ..))
메서드 이름이 get으로 시작하고 첫번째 인자값의 데이터타입이 Integer이며, 1개 이상의 인자값을 갖는 메서드 호출

execution(* com..*(..)) && @annotation(@annotation)
@annotation이 있는 모든 메소드 호출

execution(* *(..,@annotation (*), ..))
@annotation을 파라메터로 갖고 있는 모든 메소드 호출
```
<br>
@Around("execution(public * pack.BL..*
(
.
.
)
)
")를 통하여 pack.BL에서 어떤한 작업이 일어날 경우 잡아오는 역할을 한다.  
System.out.println("Hello CheckPoint1"); => 작업 수행전 실행
Object object = joinPoint1.proceed(); => 잡아온 작업 수행
System.out.println("Hello CheckPoint2"); => 작업 수행 후 실행

```java
package pack.Aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class OurAdvice {
	@Around("execution(public * pack.BL..*(..))")
	public Object kbs(ProceedingJoinPoint joinPoint1) throws Throwable {

		System.out.println("Hello CheckPoint1");
		Object object = joinPoint1.proceed();
		System.out.println("Hello CheckPoint2");
		return object;
	}
}
```
실행결과  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/AOP.JPG" height="150" width="600" /></div><br>
<br>
<hr>
내용참조:<a href="https://isstory83.tistory.com/90">I's Story 블로그</a><br>
내용참조:<a href="https://elfinlas.github.io/2017/12/14/java-annotation">MHLab 블로그</a><br>
내용참조:<a href="https://gmlwjd9405.github.io/2018/12/02/spring-annotation-types.html">gmlwjd9405 블로그</a><br>
내용참조:<a href="https://blog.naver.com/mint3081/221497953331">천프로 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Spring/tree/master/AOP">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
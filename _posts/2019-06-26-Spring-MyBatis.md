---
layout: post
title:  "Spring-MyBatis"
date:   2019-06-26 12:10:00 +0700
categories: [Spring]
---

###  Spring-MyBatis
Spring 에서 DB에 접근하기 위하여 MyBatis를 활용하였다.  
<a href="https://wjddyd66.github.io/web/2019/06/21/Web-MyBatis.html">MyBatis 설명</a>  
DB와 MyBatis를 설정하기 위하여 pom.xml에 다음과 같은 dependency를 추가하여야 한다.  
pom.xml이란 Maven(사용할 라이브러리뿐만 아니라 해당 라이브러리가 작동하는 필요한 다른 라이브러리들까지 관리하여 네트워크를 통해서 자동으로 다운받아주는 것)을 정리해 둔 곳이다.  
<a href="https://wjddyd66.github.io/spring/2019/06/25/Spring-Basic.html">Maven 설명</a>  

```xml
		<!-- Db -->
		<dependency>
			<groupId>org.mariadb.jdbc</groupId>
			<artifactId>mariadb-java-client</artifactId>
			<version>1.8.0</version>
		</dependency>

		<dependency>
			<groupId>org.springframework.boot</groupId>
			<artifactId>spring-boot-starter-jdbc</artifactId>
			<version>1.5.19.RELEASE</version>
		</dependency>

		<!-- MyBatis -->
		<dependency>
			<groupId>org.mybatis</groupId>
			<artifactId>mybatis-spring</artifactId>
			<version>2.0.0</version>
		</dependency>

		<dependency>
			<groupId>org.mybatis</groupId>
			<artifactId>mybatis</artifactId>
			<version>3.5.0</version>
		</dependency>
```

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
###  pack.resource
DB와 관련된 정보를 모아두는 곳이다.  
db.properties: DB 계정 정보 및 연결되는 DB정보  
SqlMapConfig: MyBatis가 JDBC 코드를 실행하는데 필요한 전반에 걸친 환경 설정 파일이다.  
Mapper: SQL문과 관련된 설정을 하는 파일이다.  
Configuration: Config와 Mapper를 연결시켜주는 작업이다.  
<a href="https://wjddyd66.github.io/web/2019/06/21/Web-MyBatis.html">자세한 내용</a>  
db.properties

```code
driver=org.mariadb.jdbc.Driver
url=jdbc:mysql://127.0.0.1:3306/test

username=root
password=123
```
<br>
SqlMapConfig
```java
package pack.resource;

import java.io.Reader;
import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;

public class SqlMapConfig {
  public static SqlSessionFactory sqlSession;  //DB의 SQL명령을 실행시킬 때 필요한 메소드를 갖고 있다.
 
  static{
     String resource = "pack/resource/Configuration.xml";
     try {
         Reader reader = Resources.getResourceAsReader(resource);
         sqlSession = new SqlSessionFactoryBuilder().build(reader);
         reader.close();
     } catch (Exception e) {
     System.out.println("SqlMapConfig 오류 : " + e);
  }
}
 
public static SqlSessionFactory getSqlSession(){
     return sqlSession;
  }
}
```
<br>
Configuration
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration
PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
"http://mybatis.org/dtd/mybatis-3-config.dtd">

<configuration>
 <properties resource="pack/resource/db.properties" />
 <typeAliases>
 	<typeAlias type="pack.model.JikwonDto" alias="dto"/>  
 </typeAliases>
 <environments default="dev">
  <environment id="dev">
   <transactionManager type="JDBC" />
   <dataSource type="POOLED">
    <property name="driver" value="${driver}" />
    <property name="url" value="${url}" />
    <property name="username" value="${username}" />
    <property name="password" value="${password}" />
   </dataSource>
  </environment>
 </environments>
 <mappers >
  <mapper resource="pack/resource/DataMapper.xml" />
 </mappers>
</configuration>
```
<br>
DataMapper
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper
PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
"http://mybatis.org/dtd/mybatis-3-mapper.dtd">

<mapper namespace="dev">

	<select id="selectDataAll" resultType="dto">
		select
		jikwon_no,jikwon_name,buser_name,DATE_FORMAT(jikwon_ibsail,'%Y') as
		year from jikwon,buser where buser_no=buser_num order by jikwon_no;
	</select>

	<select id="selectDataCount" resultType="string">
		select count(*) as count
		from jikwon;
	</select>

	<select id="selectDataEx1" resultType="map">
		select
		buser_name,count(*)
		as sum from jikwon,buser where buser_no=buser_num
		group by buser_num;
	</select>

	<select id="selectDataEx2" resultType="map">
		select buser_name,
		jikwon_name,max(jikwon_pay) as pay from jikwon,buser where
		buser_no=buser_num group by buser_num;
	</select>

</mapper>
```
<br>
###  pack.model
DB와 연관되는 작업을 하는 곳이다.  
JikwonDto: Jikwon정보를 전달하기 위해 만들어진 객체이다.  
 - Getter,Setter Method로 이루워져 있다.


JikwonInter: Interface를 사용하여 공동 작업시 충돌 방지  
 - List<
JikwonDto> selectDataAll(): 직원 전체 자료 조사
 - int selectDataPart(): 직원 전체 수 조사
 - List<
Map<
String, Object>> selectDataEx1(): 부서별 직원 수 조사
 - List<
Map<
String, Object>> selectDataEx2(): 부서별 가장 월급을 많이 받는 직원

JikwonImpl: 실제 DB와 연결하여 작업 DAO의 역할을 한다. 따라서 @Repository로서 Annotation을 하고 Interface에서 정의한 내용을 작성하였다.  

```java
//JikwonDto
package pack.model;

public class JikwonDto {
private String jikwon_no,jikwon_name,buser_name,year;

public String getJikwon_no() {
	return jikwon_no;
}

public void setJikwon_no(String jikwon_no) {
	this.jikwon_no = jikwon_no;
}

public String getJikwon_name() {
	return jikwon_name;
}

public void setJikwon_name(String jikwon_name) {
	this.jikwon_name = jikwon_name;
}

public String getBuser_name() {
	return buser_name;
}

public void setBuser_name(String buser_name) {
	this.buser_name = buser_name;
}

public String getYear() {
	return year;
}

public void setYear(String year) {
	this.year = year;
}

}

//JikwonInter
package pack.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.dao.DataAccessException;

public interface JikwonInter {
	List<JikwonDto> selectDataAll();
	int selectDataPart();
	List<Map<String, Object>> selectDataEx1();
	List<Map<String, Object>> selectDataEx2();
}

//JikwonImpl
package pack.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.stereotype.Repository;

import pack.resource.SqlMapConfig;

@Repository("jikwonImpl")
public class JikwonImpl implements JikwonInter {
	private SqlSessionFactory factory = SqlMapConfig.getSqlSession();

	@Override
	public List<JikwonDto> selectDataAll() {
		SqlSession sqlSession = factory.openSession();
		List<JikwonDto> list = null;
		try {
			list = sqlSession.selectList("selectDataAll");
		} catch (Exception e) {
			System.out.println("selectList Error" + e);
		} finally {
			if (sqlSession != null)
				sqlSession.close();
		}
		return list;
	}

	@Override
	public int selectDataPart() {
		SqlSession sqlSession = factory.openSession();
		int a = 0;
		try {
			String x =sqlSession.selectOne("selectDataCount");
			a=Integer.parseInt(x);
		} catch (Exception e) {
			System.out.println("selectDataPart Error" + e);
		} finally {
			if (sqlSession != null)
				sqlSession.close();
		}
		return a;
	}
	
	@Override
	public List<Map<String, Object>> selectDataEx1() {
		SqlSession sqlSession = factory.openSession();
		List<Map<String, Object>> result = null;
		try {
			result = sqlSession.selectList("selectDataEx1");
			
		} catch (Exception e) {
			System.out.println("selectList Error" + e);
		} finally {
			if (sqlSession != null)
				sqlSession.close();
		}
		return result;
	}
	
	@Override
	public List<Map<String, Object>> selectDataEx2() {
		SqlSession sqlSession = factory.openSession();
		List<Map<String, Object>> result = null;
		try {
			result = sqlSession.selectList("selectDataEx2");
			
		} catch (Exception e) {
			System.out.println("selectList Error" + e);
		} finally {
			if (sqlSession != null)
				sqlSession.close();
		}
		return result;
	}
}
```

###  pack.business
실제 Logic이 구현되는 곳이다.  
ProcessInter: Interface를 사용하여 공동 작업시 충돌 방지  
ProcessImpl: 실제 Logic이 실행되는 곳이다.  
 - selectDataAll(): 직원 전체 자료 출력
 - selectDataCount(): 직원 전체 수 출력
 - selectDataEx1(): 부서별 직원 수 출력
 - selectDataEx2(): 부서별 가장 월급을 많이 받는 직원


Main: 실제 수행을 위하여 실행을 하는 곳

```java
//ProcessInter
package pack.business;

public interface ProcessInter {
	void selectDataAll();
	void selectDataCount();
	void selectDataEx1();
	void selectDataEx2();
}


//ProcessImpl
package pack.business;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import pack.model.JikwonDto;
import pack.model.JikwonInter;

@Service
public class ProcessImpl implements ProcessInter{
	@Autowired
	@Qualifier("jikwonImpl")
	private JikwonInter inter;
	
	public ProcessImpl(JikwonInter inter) {
		this.inter=inter;
	}
	
	@Override
	public void selectDataAll() {
		List<JikwonDto> list = inter.selectDataAll();
		for(JikwonDto d: list) {
			System.out.println(d.getJikwon_no()+" "+d.getJikwon_name()+" "+d.getBuser_name()+" "+d.getYear());
		}
		
	}
	
	@Override
	public void selectDataCount() {
		int count = inter.selectDataPart();
		System.out.println("총원: "+count);
	}
	
	@Override
	public void selectDataEx1() {
		List<Map<String, Object>> list = inter.selectDataEx1();
		System.out.println(list);
		for(int i=0;i<list.size();i++) {
		System.out.print(list.get(i).get("buser_name"));
		System.out.println("  "+list.get(i).get("sum"));
		}
}
	@Override
	public void selectDataEx2() {
		List<Map<String, Object>> list = inter.selectDataEx2();
		for(int i=0;i<list.size();i++) {
		System.out.print(list.get(i).get("buser_name"));
		System.out.print("   "+list.get(i).get("jikwon_name"));
		System.out.println("   "+list.get(i).get("pay"));
		}
		
	}
}


//Main
package pack.business;

import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class Main {

	public static void main(String[] args) {
		ApplicationContext context = new ClassPathXmlApplicationContext("init.xml");
		ProcessInter inter = context.getBean("processImpl",ProcessInter.class);
		inter.selectDataAll();
		inter.selectDataCount();
		inter.selectDataEx1();
		inter.selectDataEx2();

	}

}

```
###  pack.Aspect
AOP를 사용하는 곳 이다.  
pack.business의 Method를 실행 할 때 작업을 가져와서  
시작  
작업 결과  
출력  
형식으로 만든다.  

```java
package pack.aspect;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class MyAdvice {
	@Around("execution(public * pack.business..*(..))")
	public Object kor(ProceedingJoinPoint joinPoint) throws Throwable {
		System.out.println("시작");
		Object object= joinPoint.proceed();
		System.out.println("종료");
		return object;
	}
}

```
실행결과-selectDataAll()  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis1.JPG" height="250" width="600" /></div>
<div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis6.JPG" height="250" width="600" /></div>
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis2.JPG" height="250" width="600" /></div>
<br>
실행결과-selectDataCount()  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis3.JPG" height="250" width="600" /></div>
<br>
실행결과-selectDataEx1()  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis4.JPG" height="250" width="600" /></div>
<br>
실행결과-selectDataEx2()  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/MyBatis5.JPG" height="250" width="600" /></div>
<br>
<span style ="color: red">실행결과 AOP를 적용하여 "시작"을 찍은뒤 DB에 접근하고 그 뒤 "종료"를 출력하는 것을 볼 수 있다.</span>
<br>
<hr>
참조:<a href="https://github.com/wjddyd66/Spring/tree/master/MyBatis">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
---
layout: post
title:  "Web-MyBatis"
date:   2019-06-21 09:00:00 +0700
categories: [Web]
---

###  MyBatis
객체 지향 언어인 자바의 관계형 데이터 베이스 프로그래밍을 보다 쉽게 도와주는 프레임 워크이다.  
1. SQL문이 Code로부터 완전히 분리
2. Code가 간결해짐
3. SQL문과 Code의 분리로 인한 유지보수 향상

###  DAO, DTO, FormBean
MyBatis를 이해하기 위한 기본은 DAO, DTO, Bean이다.  
 - DAO(Data Access Objects)  
    1)실질적인 DB와의 연결을 담당하는 객체  
    2)저장소에 데이터를 입력, 조쇠, 수정, 삭제 등 처리를 담당  


 - DTO(Data Tranfer Object)  
    1)데이터 전달을 위해 만들어진 객체  
    2)Private로 변수를 선언하고 getter, setter Method로 접근한다.


<a href="https://wjddyd66.github.io/java/2019/06/14/AccessModifier.html">자세한 내용</a><br>

 - FormBean:DTO와 같지만 Client 에서 WebServer로 전달하는 Data 객채이다.


DTO vs FormBean  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/dd.PNG" height="300" width="700" /></div>
<br>
그림참조:<a href="https://blog.naver.com/mint3081/221480907154">천프로 블로그</a><br>

###  MyBatis 설정
 - SqlSessionConfig.xml: MyBatis가 JDBC 코드를 실행하는데 필요한 전반에 걸친 환경 설정 파일이다.  
1)어떤 DB와 연결할 것인지  
2) 사용할 모델 클래스에 대한 별칭은 무엇인지  




```java
package pack.mybatis;

import java.io.Reader;
import org.apache.ibatis.io.Resources;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.session.SqlSessionFactoryBuilder;

public class SqlMapConfig {
  public static SqlSessionFactory sqlSession;  //DB의 SQL명령을 실행시킬 때 필요한 메소드를 갖고 있다.
//DB와 연결 
  static{
     String resource = "pack/mybatis/Configuration.xml";
     try {
         Reader reader = Resources.getResourceAsReader(resource);
         sqlSession = new SqlSessionFactoryBuilder().build(reader);
         reader.close();
     } catch (Exception e) {
     System.out.println("SqlMapConfig 오류 : " + e);
  }
}
//사용할 모델 클래스 별칭 설정
public static SqlSessionFactory getSqlSession(){
     return sqlSession;
  }
}
```
 - DataMapper: SQL문과 관련된 설정을 하는 파일이다.  
1) SQL문 id 설정  
2) ParameterType, ResultType 설정(정하지 않으면 defalut값으로 설정)  
3) SQL문 작성  




```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper
PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
"http://mybatis.org/dtd/mybatis-3-mapper.dtd">

<mapper namespace="dev">

 <select id="selectDataAll" resultType="dto">
  select * from sangdata order by code asc
 </select>
 
 <select id="selectDataById" parameterType="string" resultType="dto">
  select code,sang,su,dan from sangdata where code = #{code}
 </select>
 
 <insert id="insertData" parameterType="dto">
  insert into sangdata(code,sang,su,dan) values(#{code},#{sang},#{su},#{dan})
 </insert>
 
 <update id="updateData" parameterType="dto">
  update sangdata set sang=#{sang},su=#{su},dan=#{dan} where code=#{code}
 </update>
 
 <delete id="deleteData" parameterType="int">
  delete from sangdata where code=#{code}
 </delete>
</mapper>
```
 - Configuration.xml: Config와 Mapper를 연결시켜주는 작업이다.  
1) DB연결 정보 작성  
2)  Mapper 연결  
3)  Mapper에서 사용할 type설정(DAO 혹은 많이 사용하는 Type)  


```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration
PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
"http://mybatis.org/dtd/mybatis-3-config.dtd">

<configuration>
 <properties resource="pack/mybatis/db.properties" />
 <typeAliases>
 	<typeAlias type="pack.business.DataDto" alias="dto"/> 
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
 <mappers>
  <mapper resource="pack/mybatis/DataMapper.xml" />
 </mappers>
</configuration>
```
 - properties  
1)  .properties는 응용 프로그램의 구성 가능한 파라미터들을 저장하기 위해 자바 관련 기술, 주로 보안상의 이유로 사용한다.  
2)  DB계정은 보안이 필요한 정보이므로 properties형식에 저장하고 사용하였다.  


```xml
driver=org.mariadb.jdbc.Driver
url=jdbc:mysql://127.0.0.1:3306/test
username=root
password=123
```

###  MyBatis 예제 환경
간단한 상품 조회, 삭제, 수정, 추가 예제이다.  
아래 사진은 이번예제에서 접근하는 DB에서 Table의 정보이다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b1.JPG" height="300" width="700" /></div>
<br>
아래 코드는 접근하려는 DB의 Table의 정보에따라 DTO 객체를 만든 것이다.  
```java
package pack.business;

public class DataDto {
	private String code, sang, su, dan;

	public String getCode() {
		return code;
	}

	public void setCode(String code) {
		this.code = code;
	}

	public String getSang() {
		return sang;
	}

	public void setSang(String sang) {
		this.sang = sang;
	}

	public String getSu() {
		return su;
	}

	public void setSu(String su) {
		this.su = su;
	}

	public String getDan() {
		return dan;
	}

	public void setDan(String dan) {
		this.dan = dan;
	}
	
	
}
```

아래 코드는 DAO를 통하여 DB에 데이터를 입력, 조쇠, 수정, 삭제 등 처리를 담당하기 위한 코드이다.  
```java
package pack.business;

import java.sql.SQLException;
import java.util.List;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

import pack.mybatis.SqlMapConfig;

public class ProcessDao {
	private static ProcessDao dao = new ProcessDao();
	public static ProcessDao getInstance() {
		return dao;
	}
	
	private SqlSessionFactory factory = SqlMapConfig.getSqlSession();
	
	//전체자료를 출력하는 코드
	public List<DataDto> selectDataAll() throws SQLException{
		SqlSession sqlSession = factory.openSession(); 
		List list = sqlSession.selectList("selectDataAll");
		sqlSession.close();
		return list;
	}
	
	//일부자료를 출력하는 코드
	public DataDto selectDataPart(String code) throws Exception{
		SqlSession sqlSession = factory.openSession();
		DataDto dto = sqlSession.selectOne("selectDataById", code);
		sqlSession.close();
		return dto;
	}
	
	//자료를 넣는 코드
	public void insData(DataDto dto) throws Exception{ 
		SqlSession sqlSession = factory.openSession(true); //자동
		sqlSession.insert("insertData", dto);		
		sqlSession.close();
		
	}
	
	//자료를 업데이트 하는 코드
	public void upData(DataDto dto) throws Exception{ 
		SqlSession sqlSession = factory.openSession(true);
		sqlSession.update("updateData", dto);		
		sqlSession.close();	
		
	}
	
	//자료를 삭제하는 코드
	public boolean delData(int arg) {
		boolean b = false;
		SqlSession sqlSession = factory.openSession();		
		
		try {
			int count = sqlSession.delete("deleteData", arg);
			if(count > 0) b = true;
			sqlSession.commit();
		} catch (Exception e) {
			System.out.println("del err : " + e);
			sqlSession.rollback();
		}finally {
			if(sqlSession != null) sqlSession.close();
		}
		
		return b;
	}
	
	
}
```
###  MyBatis 예제 상품 목록 출력
아래 list.jsp는 현재 상품을 Table 형식으로 출력하는 페이지이다.  
출력되며 코드를 누르면 삭제하는 delete.jsp 실행  
이름을 누르면 업데이트하는 update.jsp 실행  
DAO(selectDataAll): 전체자료 조회 -> Mapper(selectDataAll)을 통하여 정보를 얻게 된다.  
```jsp
<%@page import="pack.business.DataDto"%>
<%@page import="java.util.ArrayList"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<jsp:useBean id="processDao" class="pack.business.ProcessDao"/>

<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
상품자료(MyBatis) <p>
<a href="ins.html">상품추가</a><br>
<%ArrayList<DataDto> slist = (ArrayList)processDao.selectDataAll(); %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<table border="1">
	<tr>
		<th>코드</th><th>품명</th><th>수량</th><th>단가</th>
	</tr>
	<c:forEach var="s" items="<%=slist %>">
	<tr>
		<td><a href="delete.jsp?code=${s.code}">${s.code}</a></td>
		<td><a href="update.jsp?code=${s.code}">${s.sang}</a></td>
		<td>${s.su}</td>
		<td>${s.dan}</td>
	</tr>	
	</c:forEach>
</table>
<b style="color: red">코드를 클릭하면 삭제, 품명을 클릭하면 수정 작업</b>
</body>
</html>
```
<br>
결과-Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b2.JPG" height="300" width="700" /></div>
<br>
결과-DB
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b3.JPG" height="300" width="700" /></div>
<br>

###  MyBatis 예제 상품 추가
아래 ins.html은 추가하려는 상품정보를 입력하는 페이지이다.  
아래 ins.jsp은 추가하려는 상품정보를 전달한다.  
DAO(insData): 상품 추가 -> Mapper(insertData)을 통하여 정보를 추가하게 된다.  
ins.jsp
```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%request.setCharacterEncoding("utf-8");%>
<jsp:useBean id="bean" class="pack.business.DataDto"/>
<jsp:setProperty property="*" name="bean"/>
<jsp:useBean id="processDao" class="pack.business.ProcessDao"/>

<%
processDao.insData(bean);
response.sendRedirect("list.jsp");
%>
```
<br>
ins.html  
```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
**상품 추가**<p/>
<form action="ins.jsp" method="post">
코드:<input type="text" name="code"><br>
품명:<input type="text" name="sang"><br>
수량:<input type="text" name="su"><br>
단가:<input type="text" name="dan"><br>
<br>
<input type="submit" value="등록">
</form>
</body>
</html>
```

<br>
결과-정보입력 Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b4.JPG" height="300" width="700" /></div>
<br>
결과-정보후 Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b5.JPG" height="300" width="700" /></div>
<br>
결과-DB
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b6.JPG" height="300" width="700" /></div>
<br>

###  MyBatis 예제 상품 삭제
아래 delete.jsp는 선택한 상품을 삭제한다.  
DAO(delData): 전체자료 조회 -> Mapper(deleteData)을 통하여 정보를 삭제 한다.  
```jsp
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<jsp:useBean id="processDao" class="pack.business.ProcessDao"/>
<%
int code = Integer.parseInt(request.getParameter("code"));
boolean b = processDao.delData(code);

if(b){
	response.sendRedirect("list.jsp");
}else{
%>
	<script>
	alert("삭제 실패");
	location.href="list.jsp";
	</script>
<%
}
%>
```
<br>
결과-삭제 전Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b5.JPG" height="300" width="700" /></div>
<br>
결과-삭제 후(황정용 Data)Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b7.JPG" height="300" width="700" /></div>
<br>
결과-DB
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b3.JPG" height="300" width="700" /></div>
<br>

###  MyBatis 예제 상품 수정
아래 update.jsp는 선택한 상품을 정보를 입력한뒤 입력한 정보에 맞게 수정한다.  
DAO(upData): 전체자료 조회 -> Mapper(updateData)을 통하여 정보를 삭제 한다.  
```jsp
<%@page import="pack.business.DataDto"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<jsp:useBean id="processDao" class="pack.business.ProcessDao"></jsp:useBean>
<%
String code = request.getParameter("code");
DataDto dto = processDao.selectDataPart(code);
%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body>
**상품 수정 **<br>
<form action="updateok.jsp" method="post">
코드 : <%=dto.getCode() %><br>
<input type="hidden" name="code" value="<%=dto.getCode() %>"><br>
품명 : <input type="text" name="sang" value="<%=dto.getSang() %>"><br>
수량 : <input type="text" name="su" value="<%=dto.getSu() %>"><br>
단가 : <input type="text" name="dan" value="<%=dto.getDan() %>"><br>
<br>
<input type="submit" value="수정">
</form>
</body>
</html>
```
<br>
결과-수정 화면 Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b8.JPG" height="300" width="700" /></div>
<br>
결과-수정 후(파인애플 Data)Web
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b9.JPG" height="300" width="700" /></div>
<br>
결과-DB
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Web/b10.JPG" height="300" width="700" /></div>
<br>

###  MyBatis include
MyBatis에서는 DataMapper에서 많이 사용하는 구문을 include하여서 사용할 수 있다.  
```xml
<!--My Batis Include-->
  <sql id="my">order by code asc</sql>

 <!-- selectDataAll과 같은 구문 -->
 <select id="selectDataAll2" resultType="dto">
  select * from sangdata
  <include refid="my1"/>
 </select>
```
<br>
<hr>
자료참조:<a href="https://m.blog.naver.com/PostView.nhn?blogId=wwwkang8&logNo=220989381100&proxyReferer=https%3A%2F%2Fwww.google.com%2F">강정호 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Web/tree/master/mybatis_web">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
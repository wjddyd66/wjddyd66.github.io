---
layout: post
title:  "JavaScript-Suggest"
date:   2019-06-20 08:30:00 +0700
categories: [Web]
---

###  Suggest

Suggest란 검색어를 입력하면 자동으로 검색어를 추천하는 시스템이다.  
아래 예제는 DB에 있는 내용을 검색해주는 Suggest기능을 구현한 예제이다.  
아래 사진은 현재 DB에 있는 자료이다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/JavaScript/Js33.JPG" height="150" width="600" /></div><br>

###  Html
1. 실제 행동을 적용시키는 Suggest.js 연결
2. Input,Output 나올 곳 정의
 - Input: Value(이름을 검색하는 곳 이다.) Text입력시 suggest.js의 sijak()실행
 - Output
  - Suggest(Db에 있는 자료를 검색해 보여주는 곳)
  - Selected Name(사용자가 선택한 결과를 보여주는 곳)

```html
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Suggest</title>

<!-- Script -->
<script src="suggest.js">

</script>

</head>
<body>
*Name Search By Ajax<p/>
<form name="frm">
    <!-- 검색어를 입력하는 곳 값이 바뀌면 suugest.js 의 sijak()함수 실행-->
    Value: <input type="text" name="keyword" id="keyword" onkeydown="sijak()">
    <!-- Suggest 결과가 나오는 곳 -->
    <div id="suggest" style="display: position:absolute; lefr:100px; top:30px;"></div>
    <hr>
    <!-- 사용자가 선택한 결과가 나오는 곳 -->
    Selected Name: <input type="text" name="sel" size="10" readonly="readonly">
</form>
</body>
</html>
```
###  JavaScript
1. sijak(): Value에 값이 생기면 실행하는 함수.  
1초뒤 sendKeyword() 실행  
1초의 텀을 둔 이유는 한글에서 한 글자를 적을때까지 기달리게 하기 위해서 이다.

2. sendKeyWord(): Ajax를 활용하여 suggest.jsp에게 요청을 하는 함수이다.  
넘겨주는 값은 keyword이며 Post방식으로 전송한다.

3. process(): Ajax의 결과를 받고 처이하는 공간이다. Ajax에게 받은 Data(이름)에 링크를 건뒤 Suggest에 보여지는 형식이다.

4. func(): 이름 클릭시 Selected에 값을 넣는 함수이다.
5. hide(): Suggest창을 숨기는 함수 이다.
6. show(): Suggest창이 보이게 속성을 바꾸는 함수이다.


```javascript
var xhr;
var checkFirst = loopSend = false;
var lastKeyword = "";
//Timeout을 1초로 걸었다. 일정 단어 완성 뒤 검색어를 만들어 검색하기 위해서 이다.
function sijak() {
    if (checkFirst == false) {
    	//1초뒤 sendKeyword() 수행
        setTimeout("sendKeyword()", 1000);
        loopSend = true;
    }
}

//Ajax를 활용하여 suggest.jsp 에게 요청을 하는 함수이다. 넘겨주는 값은 keyword이며 Post방식으로 전송한다.
function sendKeyword() {
    if (loopSend == false)
        return;
    else {
        var keyWord = document.frm.keyword.value;
        //키워드가 hide 함수를 불러 검색어 창 숨기기
        if (keyWord === "") {
            lastKeyword = "";
            hide("suggest");
            //검색어가 있는 경우 suggest.jsp에게 값 요청
        } else if (keyWord !== lastKeyword) {
            lastKeyword = keyWord;

            var para = "keyword=" + keyWord;
            xhr = new XMLHttpRequest();
            xhr.open("post", "suggest.jsp", true);
            xhr.onreadystatechange = function () {
                if (xhr.readyState == 4) {
                    if (xhr.status == 200) {
                        process();
                    } else {
                        alert("요청실패1" + xhr.status)
                    }
                }
            }
            xhr.setRequestHeader("Content-Type", "application/x-www-form-urlencoded");
            xhr.send(para);


        }
    }
}

//Ajax의 결과를 받고 처리하는 공간이다.
//Ajax에게 받은 Data(이름)에 링크를 건 뒤 Output-Suggest에 보여지는 형식이다.
function process() {
    var data = xhr.responseText;
    var result = data.split("|");
    var tot = result[0];
    if (tot > 0) {
        var datas = result[1].split(",");
        var imsi = "";
        //각각의 이름에 링크 걸기 각각의 이름은 func(자기이름)이 들어가 있다.
        for (var i = 0; i < datas.length; i++) {
            imsi += "<a href=\"javascript:func('" + datas[i] + "')\">" + datas[i] + "</a><br>";
        }
        //Output-Suggest에 결과 보여주기
        var listView = document.getElementById("suggest").innerHTML = imsi;
    }

}

//이름 클릭시 Output-Selected 에 값 넣기. Suggest창 숨기기
function func(reData) {
    frm.sel.value = reData;
    loopSend = checkFirst = false;
    lastKeyword = "";
    hide("suggest");

    frm.keyword.value="";

}

//Suggest 창 숨기는 함수
function hide(ele) {
    var e = document.getElementById(ele);
    if (e) e.style.display = "none";
}

//Suggest 창 보이게 속성 바꾸는 함수
function show(ele) {
    var e = document.getElementById(ele);
    if (e) e.style.display = "";
}
```

###  Jsp
Jsp(JavaServer Page): HTML 코드에 JAVA코드를 넣어 동적 웹페이지를 생성하는 웹어플리케이션 도구 이다.  
Jsp에서는 DB에 연결하여 Query문을 날린다음에 결과를 반환한다.  
Query: select jikwon_name from jikwon where jikwon_name like ?  
?는 suggest.js에서 받은 keyword이다.  
```jsp
<%@page import="java.util.ArrayList"%>
<%@page import="java.sql.DriverManager"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
	pageEncoding="UTF-8"%>
<!-- Jsp에서는 DB에 연결하여 Query문을 날린다음에 결과를 반환한다. -->
<!-- Query: select jikwon_name from jikwon where jikwon_name like ? -->
<!-- ?는 suggest.js에서 받은 keyword이다. -->
<%
	request.setCharacterEncoding("utf-8");
	String keyword = request.getParameter("keyword");

	Connection conn = null;
	PreparedStatement pstmt = null;
	ResultSet rs = null;
	String result = "";

	try {
		Class.forName("org.mariadb.jdbc.Driver");
	} catch (Exception e) {
		System.out.println("Connection Error: " + e);
		return;
	}

	try {
		conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/test", "root", "123");
		pstmt = conn.prepareStatement("select jikwon_name from jikwon where jikwon_name like ?");
		pstmt.setString(1, keyword + "%");
		rs = pstmt.executeQuery();

		ArrayList<String> list = new ArrayList<String>();
		while (rs.next()) {
			list.add(rs.getString(1));
			//System.out.println(rs.getString(1));
		}

		out.print(list.size());
		out.print("|");
		for (int i = 0; i < list.size(); i++) {
			String data = list.get(i);
			out.print(data);
			if (i < list.size() - 1) {
				out.print(",");
			}
		}
		
		rs.close();
		pstmt.close();
		conn.close();
	} catch (Exception e) {
		System.out.println("Process Error: " + e);
		return;
	}
%>
```
<br>
결과: 
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/a30f391d1b0c4bc78cfc08743fa6d382" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

<hr>
내용참조:<a href="https://webclub.tistory.com/218">Web Club 블로그</a><br>
내용참조:<a href="http://tcpschool.com/jquery/jq_event_delegation">TCP School</a><br>
내용참조:<a href="https://coding-factory.tistory.com/143">코딩팩토리 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Web/tree/master/Suggest">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
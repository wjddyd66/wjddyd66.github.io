---
layout: post
title:  "JAVA-Interface"
date:   2019-06-14 12:50:20 +0700
categories: [JAVA]
---

### 자바의 인터페이스: 공동 작업시 충돌을 방지하기 위하여 작성
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">

	<tr bgcolor="silver">	
		<th>Parent Type</th>
		<th>Format</th>
		<th>사용 이유</th>
		<th><span style ="color: red">다중 상속</span></th>
	</tr>
	
	<tr>
		<td>Class</td><td>Child extends Parent</td><td>공통된 특성 재사용</td><td>O</td>
	</tr>
	
	<tr>
		<td>Interface</td><td>Child implements Parent</td><td>공동 작업시 충돌 방지</td><td>X</td>
	</tr>

</table>
<br><br>

공동 작업시 충돌 방지라는 것은 변수와 Method를 미리 선언하여 변경 불가하게 만든다는 뜻이다. 변수의 경우 Final Type으로 선언하여 값을 변경 불가능 하게 한다.  
<span style ="color: red">**Type 앞에 붙일 수 있는 것은 Static과 Final이 존재한다.**</span>

1. Static: 메모리에 고정하여 모든 객체가 공유하는 자원

2. Final: 변수가 한번 선언되면 변경 불가능한 자원  

Method의 경우 선언만 해주고 내용은 물려받은 Child에서 정의하게 된다.  
<span style ="color: red">**정의한 Method를 Overriding 하지 않으면 Error가 나오게 된다.  이러한 이유로 실제 Project에서 Project 관리하는 사람이 꼭 필요한 기능을 정의하고 팀원들에게 배포하는 형식을 취할 때 많이 사용하게 된다.**</span>

{% highlight java %}
//Parent.java
public interface Parent {
	/*
	Interface 에서 변수선언은 final 로서 변하지 않게 선언해 주어야 한다.
	final 로 선언하게 되면 값이 변하지 않는 특성을 가지게 된다.
	*/
	public final String name="Kim";
	
	//자바에서 Method는 내용을 선언하지 않고 선언만 해준다.
	public void setName(String name);
	public String getName();
}

//Parent2.java
public interface Parent2 {
	/*
	Interface 에서 변수선언은 final 로서 변하지 않게 선언해 주어야 한다.
	final 로 선언하게 되면 값이 변하지 않는 특성을 가지게 된다.
	*/
	public final String name="Kim";
	
	//자바에서 Method는 내용을 선언하지 않고 선언만 해준다.
	public void setName(String name);
	public String getName();
}

//Child.java
/*
Interface를 상속받는 경우는 implements로서 상속받게 된다.
Interface의 장점은 다중 상속이 가능하다.
 */
public class Child implements Parent,Parent2{
	String name;
	
	public static void main(String[] args) {
		System.out.println("Interface");
		//Parent.name="Hwang" Parent에서 Final로 정의되어있어 값을 변경 할 수 없다.
		System.out.println(Parent.name);
		//Interface, Kim
	
	}


​	
​	//Interface에서 정의한 Method는 반드시 Child Class에서 정의하여야 한다.
​	@Override
​	public String getName() {
​		return name;
​	}
​	
​	@Override
​	public void setName(String name) {
​		this.name = name;
​	}

}
{% endhighlight %}
<br>

<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/Interface">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


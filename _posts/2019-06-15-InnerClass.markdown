---
layout: post
title:  "JAVA-내부클래스, 익명클래스"
date:   2019-06-15 10:15:20 +0700
categories: [JAVA]
---

### 내부 클래스

코드의 복잡성과 가독성을 높이기 위하여 사용한다.  
장점  

1. 작성하는 데 더 적은 코드가 요구된다.
2. 클래스 내부에 존재하기 때문에 가독성이 있고 유지 보수 가능한 개발 가능
3. 외부 클래스의 모든 멤버에 접근할 수 있다는 장점
   {% highlight java %}

	/*
	InnerClass.java
	OuterClass - TestInnerClass
	InnerClass - Inner
	내부클래스: 코드의 복잡성과 가독성을 높이기 위하여 사용한다.
	외부클래스의 멤버에 쉽게 접근 가능하다는 장점을 가지고 있다.
	*/
	public class InnerClass {
		
		//Outer Class의 Value
		String outer_value ="Hello";
		
		//Outer Class의 Method
		public void outMethod() {
			System.out.println("OuterMethod");
		}
		
		//Inner Class 선언
		public class Inner{
			//Inner Class의 Value
			String inner_value ="World";
			//Inner Class의 Method
			public void innerMethdod() {
				System.out.println("InnerMethod");
			}
		}
		
		public static void main(String[] args) {
			//OuterClass 선언 및 객체화
			InnerClass outer = new InnerClass();
			//InnerClass는 OuterClass.new InnerClass()로서 객체화 하여 사용
			Inner inner = outer.new Inner();
			
			outer.outMethod();
			inner.innerMethdod();
			System.out.println(outer.outer_value+inner.inner_value);
			/*
			OuterMethod
			InnerMethod
			HelloWorld
			 */
		}
	}
{% endhighlight %}
<br>

### 익명 클래스
<span style ="color: red">**클래스의 선언과 객체의 생성이 동시에 되는 클래스 이다. 1개의 객체만을 생성하고 1번만 사용되는 특징을 가지고 있다.**</span>    
클래스를 새로 하나 구현하는 것이 큰 비용(Time, Memory)이 소모 될때 쓴다.
Format: "ClassName" ObjectName = new "ClassName"(){Member Field};  

<h3>내부 익명 클래스</h3>

<span style ="color: red">**추상 클래스인 경우 바로 객체로 선언할 수 없다.**</span><br><span style ="color: red">**내부 클래스 + 익명 클래스를 활용한 내부익명 클래스 선언=>추상 클래스를 바로 객체화 하여 사용할 수 있다.**</span>



{% highlight java %}


	/*
	Person.java
	Abstract 선언
	*/
	abstract class Person {
		abstract void Name();
		abstract void Age(); 
	}
	
	/*
	AnonymousClass.java
	내부 익명 클래스 구현
	*/
	public class AnonymousClass {
	public static void main(String[] args) {
		//익명 클래스 구현 Interface이므로 Method를 Override하여 구현해야 한다.
		Person Hwang = new Person() {
			@Override
			public void Name() {
				System.out.println("황정용");	
			}
			@Override
			public void Age() {
				System.out.println("26");	
			}
		};
		Hwang.Name();
		Hwang.Age();
		/*추상 클래스인 경우 바로 객체로 선언할 수 없다.
		내부 클래스 + 익명 클래스를 활용한 내부익명 클래스 선언
		=>추상 클래스를 바로 객체화 하여 사용할 수 있다.
		*/
	}
	}

{% endhighlight %}
  <br>

<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/InnerClass">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


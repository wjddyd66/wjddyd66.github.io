---
layout: post
title:  "JAVA-상속"
date:   2019-06-14 12:45:20 +0700
categories: [JAVA]
---

### 자바의 상속  
Class 를 통하여 Object를 만들 수 있다는 것을 공부하였다.  
Object들 중 똑같은 변수 혹은 Method를 공통으로 가지고 있는 객체들이 있다면 Parent Class에 한 번만 정의하여 계속해서 선언해야 하는 불편함을 감소 시킬수 있다.  
Format: 자식클래스  <span style ="color: red">extends</span>  부모 클래스  
<span style ="color: red">자바에서는 다중 상속이 불가능 하다.</span>

### Overriding , Overloading
Overriding 과 Overloading의 경우 용어가 비슷하여 많이 혼동된다. 하지만 용어만 비슷할 뿐 개념은 아예 다르므로 비교하여 알아두자.
1. Overriding: 상속에서 Child Class는 Parent Class의 변수 혹은 Method를 사용할 수 있다고 하였다. Overriding은 이러한 Parent Class 의 Method 를 Child Class에서 변경하는 것을 의미한다.   @Override로서 Overriding을 표시하는 것을 권장한다.  
<span style ="color: red">Annotaion: AOP를 편리하게 구성하기 위하여 사용 @~ 로서 표현한다.</span> 

2. Overloading: 같은 클래스 내 에서 같은 이름의 Method를 사용하는 것 이다.
	1. Argument 의 개수가 다르다.
	2. Argument 의 Type이 다르다.
    => 목적이 같은 Method이나 Type이나 개수에 따라서 달라지는 Method를 선언할 때 적합

    {% highlight java %}
//다중 상속을 위한 부모 선언
public class MultipleInheritance {
	public void msg() {
		System.out.println("Hello World");
	}
}
    //Parent.java 부모 선언
    public class Parent {
    	private String name;
    /*
    오버로딩(Overloading): 같은 클래스 내 에서 같은 이름의 Method를 사용하는 것 이다.
  1. Argument 의 개수가 다르다.
  2. Argument 의 Type이 다르다.

    => 목적이 같은 Method이나 Type이나 개수에 따라서 달라지는 Method를 선언할 때 적합
    */
    public void setName(String name) {
    	this.name = name;
    }

  public void setName(String name,String name2) {
  	this.name = name;
  }

  public void setName(String name,int name2) {
  	this.name = name;
  }

  public String getName() {
  	return name;
  }
  }
  //Child.java 자식선언
  /*
  Format: 자식클래스 extends 부모클래스로 자식이라는 것을 표현
  자바에서는 다중 상속이 불가하다.
  public class Child extends Parent,Multiplenheritance 불가
  */
  public class Child extends Parent{
  	String name;
  	
  /*Annotaion: AOP를 편리하게 구성하기 위하여 사용
  @Override: 메소드가 오버라이드 됬는지 검증
  Overriding: 부모에서 정의한 것을 자식 Class에서 내용을 변경해야 할 상황이
  올때 사용할 수 있다.
  */
  @Override
  public void setName(String name,int name2) {
  	this.name = name;
  }

  //자식 클래스는 부모클래스에 선언된 것을 사용할 수 있다.
  public static void main(String[] args) {
  	Child c1 = new Child();
  	Child c2 = new Child();
  	
  	//부모 클래스에 public으로 선언된 set,get Method에 접근 가능하다.
  	c1.setName("Tom");
  	c2.setName("James");
  	c1.getName();
  	c2.getName();
  	
  	Parent p1 = new Child();
  	/*부모는 자식으로서 선언 가능하다.
  	Child c3 = new Parent();
  	Error => 자식은 부모로서 선언 불 가능하다.
  	자식에서 Overriding으로 더 선언 되어 있을 수 있기 때문이다.
  	*/

  }
  }

{% endhighlight %}
<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/Inheritance">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.




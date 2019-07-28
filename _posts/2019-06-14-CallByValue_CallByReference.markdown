---
layout: post
title:  "JAVA-CallByValue, CallByReference"
date:   2019-06-14 12:10:20 +0700
categories: [JAVA]
---

### 변수의 저장 방식  
CallByValue, CallByReference의 설명의 들어가기 앞서 컴퓨터의 값의 저장방식에 대해서 알아야 한다. 밑의 그림을 보면 컴퓨터이 저장 방식을 알 수 있다.

<img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/_posts/Reference.PNG" height="300" width="600" />

참조 :<a href="https://dojang.io/mod/page/view.php?id=509">코딩도장</a>

1. 주소 - 저장되어있는 공간
2.  값 - 저장되어있는 공간에 가지고 있는 값

밑의 코드를 보게 되면 변수가 선언되는 과정을 알 수 있다.

{% highlight java %}
int a=3;
//변수를 선언하면 자동으로 주소를 할당하고 값을 넣게 된다.
System.out.println(a);
//저장한 변수에는 선언한 변수의 이름으로 접근하게 된다.
//int a => 저장 공간은 4Byte로 선언하고 4Byte의 공간에 값 3 을 넣게 된다.
{% endhighlight %}

### CallByValue, CallByReference  

1. Call By Value : 호출 시 값을 복사하여 준다.
2. Call By Reference: 호출 시 주소를 복사하여 준다.



<span style ="color: red">**결론부터 말하자면 자바는 Call By Value 방식이다. 주소로서 접근하는 것이 아닌 값으로서 접근하게 된다. 밑의 코드는 Call By Value 방식과 Call By Reference는 아니지만 Call By Reference 같이 접근하는 방식의 Code이다.**</span>   

{% highlight java %}
public class CallByValue_CallByReference {
	String s;
	
public CallByValue_CallByReference(String s) {
	this.s=s;
}

public static void main(String[] args) {
	//Call By Value 의 예제이다
	String s1 = "Hello";
	String s2 = "World";
	ValueSwap(s1, s2);
	System.out.println(s1+s2);
	/*
	WorldHello
	HelloWorld
	결과에서 보았듯이 주소가 아닌 값을 참조하여 s1과 s2가 Swap되지 않는 결과를 보인다.
	*/
	CallByValue_CallByReference ss1 
	= new CallByValue_CallByReference("Hello");
	
	CallByValue_CallByReference ss2 
	= new CallByValue_CallByReference("World");
	ReferenceSwap(ss1,ss2);
	System.out.println(ss1.s+ss2.s);
	/*
	WorldHello
	WorldHello
	객체의 값을 바꾼 것 이기 때문에 값이 Swap되었다.
	하지만 Call By Reference는 아니고 Call By Value로서
	Call By Reference처럼 흉내를 낸 것이다.
	객체안의 변수에 접근하여 값을 바꾼 것 이기 때문에 위의 결과와 다르게 나오게 된다.
	*/
}

//Call By Value Method
static void ValueSwap(String one, String two) {
	String temp = one;
	one = two;
	two = temp;
	System.out.println(one+two);
}

//Call By Reference같은 Method
static void ReferenceSwap(CallByValue_CallByReference ss1, CallByValue_CallByReference ss2) {
	String temp =ss1.s;
	ss1.s =ss2.s;
	ss2.s =temp;
	System.out.println(ss1.s+ss2.s);
}
}
{% endhighlight %}
<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/blob/master/Basic/CallByValue_CallByReference.java">원본코드</a><br>
참조:<a href="https://sleepyeyes.tistory.com/11">sleepyeyes 블로그</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
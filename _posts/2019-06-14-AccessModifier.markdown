---
layout: post
title:  "JAVA-접근제어자"
date:   2019-06-14 08:45:20 +0700
categories: [JAVA]
---

### 자바의 접근제어자: Member또는 Class에 해당되는 Member 또는 클래스를 외부에서 접근하지 못하도록 제한하는 역할을 함
### 자바의 접근 제어자
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">

	<tr bgcolor="silver">	
		<th>접근 제어자</th>
		<th>범위</th>
	</tr>
	
	<tr>
		<td>Private</td><td>Class내</td>
	</tr>
	
	<tr>
		<td>Default</td><td>같은 패키지</td>
	</tr>
	
	<tr>
		<td>Protected</td><td>같은 패키지+ 다른 패키지의 자손 Class</td>
	</tr>
	
	<tr>
		<td>Public</td><td>모든 범위</td>
	</tr>

</table>
<br>

### 1. Private
Private는 Class내에서만 접근할 수 있어서, 자원이나 Logic을 보호하기 위하여 사용 된다  
<span style ="color: red">**Encapsulation: 보호된 자원에 접근하기 위하여 Set, 과 Get Method를 활용하여 자원의 변경을 한다. => Encryption과 Automation의 장점이 있다.**</span>

<span style ="color: red">**Get 과 Set을 사용할때 같은 같은 변수명을 사용하기 위하여 this를 사용하게 된다. this는 객체, 자기 자신을 나타내게 된다. => 변수의 이름을 같게하여 자원의 재활용성을 늘리게 된다.**</span>

{% highlight java %}
	private int a = 3;
	//Public - Variables and methods with public 
	access control can be accessed from any Class
	public int getA() {
		return a;
	}
	public void setA(int a) {
		this.a = a;
		/*
		 this. is used to make sure that the global field is 	  the instance field 
		 when the parameters of the method or constructor are    	  the same.
		 =>Increase recyclability by continuing to use 		  	   variables of the same name
		 */
	}
	
	/*
	 Encapsulation - To access the private variable, 
	 access the method as public and execute the operation on the variable.
	 Advantage: Encryption, Automation
	 */
{% endhighlight %}

### 2. Protected
Protected가 붙은 변수, 메소드는 동일 패키지내의 클래스 또는 해당 클래스를 상속받은 외부 패키지의 클래스에서 접근이 가능하다.   
{% highlight java %}		

	public class Car {
		protected String name = "Car";
	}
	
	public class Sonata {
		public static void main(String[] args) {
			Car c1 = new Car();
			System.out.println(c1.name);
			//car
		}
	}
{% endhighlight %}
### 3. Default

Access Modifier를 설정하지 않으면 자동으로 정의되는 Access Modifier이다. 해당 패키지 내에서만 접근이 가능하다.

<br>
{% highlight java %}

	int a =3;
{% endhighlight %}
<br>

### 4. Public

Public은 Public 접근 제어자가 붙은 변수, 메소드는 어떤 클래스 에서도 접근이 가능하다.

<span style ="color: red">**Public으로 선언하게 되면 어디에서도 사용가능한 편리성이 생기지만, Private의 장점인 Encryption과 Automation를 보장받을 수 없다. => 많이 사용되는 것만 최소한으로 사용하는 습관을 들이도록 하자 **</span>

<br>
{% highlight java %}

```
public class Car {
	protected String name = "Car";
}

public class Sonata {
	public static void main(String[] args) {
		Car c1 = new Car();
		//위에서 Car라는 Class를 public으로 선언하여 
		다른 클래스에서 접근하여 객체를 만들 수 있었다.
	}
}
```

{% endhighlight %}
<br>

<hr>
참조: <a href="https://github.com/wjddyd66/JAVA/tree/master/AccessModifier">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


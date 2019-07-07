---
layout: post
title:  "Spring-Project-스케줄러"
date:   2019-06-27 09:30:00 +0700
categories: [Project]
---

###  스케줄러
스케줄러란 특정 Job을 특정시간에 처리하기 위한 Service이다.  
비행기 티켓의 정보는 매일매일 DB에 사용자가 사용한 만큼 추가된다.  
하지만, 비행기가 도착하고 나서는 필요없는 정보가 된다.  
따라서 매일 일정시간에 필요없어진 정보를 삭제하기 위하여 스케줄러를 사용하게 되었다.  

스케줄러를 사용하기 위하여 Annotaion으로 스케줄러인것을 명시한 다음 cron이란 변수에 몇시에 시작할 것인지 선언한다.  
```code
@Scheduled(cron = "0 0 3 * * *")  
```
<br>
cron 표현식(왼쪽부터 오른쪽 순으로 다음과 같은 의미가 있다.)  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>단위</td><td>사용 가능한 범위</td>
	</tr>
	<tr>
		<td>Minutes</td><td>0~59</td>
	</tr>
		<tr>
		<td>Hours</td><td>0~23</td>
	</tr>
		<tr>
		<td>Day of Month</td><td>1~31</td>
	</tr>
			<tr>
		<td>Month</td><td>1~12</td>
	</tr>
			<tr>
		<td>Day of Week</td><td>1~7(1: 일요일, 7: 토요일)</td>
	</tr>
			<tr>
		<td>Years(optional)</td><td>1970~2099</td>
	</tr>
	</tbody>
</table>
<br>
사용 특수문자의 사용은 아래와 같은 의미가 있다.  
<table class="table">
	<tbody>
	<tr>
		<td style="width:50px">특수문자</td><td>의미</td>
	</tr>
	<tr>
		<td>*</td><td>모든수를 의미, Minutes 위치에 사용될 경우 매분마다 라는 뜻</td>
	</tr>
		<tr>
		<td>?</td><td>Day of Month, Day of Week에만 사용 가능, 특별한 값이 없다는 뜻</td>
	</tr>
		<tr>
		<td>,</td><td>특정 시간을 설정. Day of Week 위치에 2, 4, 6 이라고 쓰면 월, 수, 금에만 동작하라는 뜻</td>
	</tr>
			<tr>
		<td>/</td><td>증가를 표현, Seconds 위치에 0/15로 설정되어 있으면, 0초에 시작해서 15초 간격으로 동작 하라는 뜻 </td>
	</tr>
			<tr>
		<td>L</td><td>Day Of Month 에서만 사용하며, 마지막 날의 의미 Day of Month 에 L로 설정되어 있으면 그달의 마지막날에 실행하라는 의미</td>
	</tr>
			<tr>
		<td>W</td><td>Day of Month 에만 사용하며, 가장 가까운 평일을 의미. 15W로 설정되어 있고 15일이 토요일이며, 가장 가까운 평일인 14일 금요일에 실행, 15일이 일요일이면 16일 월요일에 실행된다.15일이 평일이면 그날 그대로 실행됨</td>
	</tr>
				<tr>
		<td>LW</td><td>Day of Week에 사용, 6#3 의 경우 3번째 주 금요일에 실행된다.</td>
	</tr>

	</tbody>
</table>
<br>
스케줄러를 사용하게 되면 기존 작업에서 새로운 작업을 하게 되므로 서버에 부하가 걸리게 된다.  
매일매일 작동해야 하므로 사용자의 수가 적은 새벽3시와 4시에 나누어서 필요없는 작업을 삭제하게 구성하였다.  

스케줄러를 위한 xml 추가
```xml
	<task:scheduler id="jobScheduler" pool-size="10" />
	<task:annotation-driven scheduler="jobScheduler" />
```
<br>
스케줄러 Controller
```java
@Component
public class Scheduler {




	@Autowired
	private SchedulerInter inter;

	@Scheduled(cron = "0 0 3 * * *")
	public void DeleteTableADaily() {
		Date today = new Date();
		System.out.println("현재 시간 : " + today);
		System.out.println("스케줄러 실행 - 이륙 날짜가 전 날인 노선 삭제");
		String tname = "<CURDATE()";
		inter.DailyDelete(tname);
	}

	@Scheduled(cron = "0 0 4 * * *")
	public void DeleteTableADaily2() {
		Date today = new Date();
	    System.out.println("현재 시간 : " + today);
		System.out.println("스케줄러 실행 - 전날 비행기 객석 정보 삭제");
		for(int i=1;i<=9;i++) {
			for(int j=1;j<=9;j++) {
				String sql="delete from ba"+i+"0"+j+" where 			  			STR_TO_DATE(substring(t_no,2,8),'%Y%m%d') < CURDATE()";
				String sql2="delete from ba"+i+"0"+j+"a where 						STR_TO_DATE(substring(t_no,2,8),'%Y%m%d') < CURDATE()";
				String sql3="delete from ba"+i+"0"+j+"b where 						STR_TO_DATE(substring(t_no,2,8),'%Y%m%d') < CURDATE()";
				String sql4="delete from ba"+i+"0"+j+"c where 						STR_TO_DATE(substring(t_no,2,8),'%Y%m%d') < CURDATE()";
				String sql5="delete from ba"+i+"0"+j+"d where 						STR_TO_DATE(substring(t_no,2,8),'%Y%m%d') < CURDATE()";
				inter.DailyDelete2(sql);
				inter.DailyDelete2(sql2);
				inter.DailyDelete2(sql3);
				inter.DailyDelete2(sql4);
				inter.DailyDelete2(sql5);
			}
		}
	}
}
```
실제 스케줄러 작동 전 후 DB 사진  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Sc.PNG" height="400px" width="350px" /></div><br>

<br>
<hr>
내용참조:<a href="https://chochochobodeveloper.tistory.com/5">초초초초보개발자 블로그</a><br>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
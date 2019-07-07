---
layout: post
title:  "Spring-Project-공지사항"
date:   2019-06-27 07:30:00 +0700
categories: [Project]
---

###  공지사항-DB 구성
공지사항을 위한 DB구성은 아래와 같다.  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE1.PNG" height="100%" width="100%" /></div><br>

###  공지사항-Pgae 구성
공지사항의 Page는 크게 Header, Body, Bottom으로 구분되어있다.  
Header 와 Bottom 같은경우 Main Page와 같은 구성으로 되어있다.  
공지사항의 Page의 Body같은 경우 Session의 id값을 비교하여 id가 admin 즉, 관리자 이면 새글쓰기, 기존 글 수정, 삭제가 보이도록 구성하였다.  
Session의 id값을 비교하여 관리자를 알아내는 것은 아래 코드로 구현을 하였다.  

```jsp
		<!-- 게시글에 관리자 권한 에따라 목록 보이기 -->
		<table border="1" class="table" style="margin-top: 100px">
			<tr style="background-color: silver; color: black;">
				<th>번호</th>
				<th>제목</th>
				<th>조회수</th>
				<th>날짜</th>
				<c:if test="${id eq 'admin' }">
					<th colspan="2">관리자 권한</th>
				</c:if>
			</tr>
			<c:forEach var="s" items="${list}">
				<tr>
					<td>${s.num }</td>
					<td
						onClick="location.href='gong_detail?num=${s.num}&spage=<%=request.getParameter("spage")%>&sword=${sword}'"
						style="cursor: pointer; width: 500px;">${s.title }</td>
					<td>${s.readcnt }</td>
					<td>${s.bdate }</td>
					<c:if test="${id eq 'admin' }">
						<td><a
							href="gong_update?num=${s.num}&spage=<%=request.getParameter("spage")%>&sword=${sword}">수정</a></td>
						<td><a href="#" onclick="delchk(${s.num },'${sword}')">삭제</a></td>
					</c:if>
				</tr>
			</c:forEach>
		</table>

<!-- 게시글에 관리자 권한 에따라 새글 쓰기 추가 -->
		<div style="text-align: center; margin-bottom: 120px;">
			<form action="gong_list?spage=1" name="frm" method="post">
				<input type="text" name="sword"> <input type="submit"
					class="btn btn-outline-dark" value="검색" id="btnSearch">
				<c:if test="${id eq 'admin' }">
					<a class='nav-link' href='gong_write'>새글 추가</a>
				</c:if>
			</form>
		</div>
	</section>
```
일반 User 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE2.PNG" height="250" width="600" /></div><br>
<br>
Admin 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/GE4.PNG" height="250" width="600" /></div><br>
<br>

###  공지사항 Pagination,검색어
페이징이란 게시판을 page 단위로 나누는 행위를 의미한다.  
페이지를 옮기거나 검색어가 있을경우 그에 맞는 페이지를 보여주어야 함으로   
spage = 현재 페이지수  
sword = 검색어  
로 정의하고 코드를 작성하였다.  
Controller에서 전체 페이지와 페이지당 출력 자료 수를 정하여 Paging처리하는 코드는 아래와 같다.  
```java
	private int tot; // 전체 레코드 수
	private int pList = 12; // 페이지 당 출력 자료 수
	private int pageSu; // 전체 페이지 수
	private int blocksu; // 전체 Block 을 몇개로 나눌지

	@Autowired
	@Qualifier("gongImpl")
	private gongImpl inter;

	//글 전체보기
	@RequestMapping("gong_list")
	private ModelAndView process(@RequestParam("spage") String spage, HttpServletRequest request) {
		HashMap<String, Object> map = new HashMap<String, Object>();
		if (request.getParameter("sword") == null)
			map.put("sword", "All");
		else
			map.put("sword", request.getParameter("sword"));
		
		tot = inter.Pagesu(map);

		if (tot % pList == 0)
			pageSu = tot / pList;
		else
			pageSu = tot / pList + 1;

		int page = Integer.parseInt(spage);

		if (page % 5 != 0)
			blocksu = page / 5 + 1;
		else
			blocksu = page / 5;

		List<gongDto> list = inter.selectList(map);
		int k = 0;
		ArrayList<gongDto> list2 = new ArrayList<gongDto>();
		try {
			while (k < pList) {
				gongDto dto = new gongDto();
				dto.setNum(list.get(((page - 1) * pList) + k).getNum());
				dto.setTitle(list.get(((page - 1) * pList) + k).getTitle());
				dto.setBdate(list.get(((page - 1) * pList) + k).getBdate());
				dto.setReadcnt(list.get(((page - 1) * pList) + k).getReadcnt());
				list2.add(dto);
				k++;
			}
		} catch (Exception e) {
			System.out.println("페이지수 예외처리" + e);
		}

		ModelAndView view = new ModelAndView();
		view.setViewName("gong_main");
		view.addObject("list", list2);
		view.addObject("su", pageSu);
		view.addObject("bsu", blocksu);
		view.addObject("sword", map.get("sword"));
		return view;
	}
```
<br>
View에서 사용자에게 보여지는 부분이다.  
Controller를 거쳐 나온 변수를 가지고 페이징을 완성하는 코드이다.  
다음이나 이전을 눌렀을 경우 마지막 페이지거나 첫번째 페이지의 경우 더 이상 이동하면 안됨으로 예외처리를 적용하였다.  
```jsp
		<!-- 페이징 처리 -->
		<div class='d-flex'>
			<ul class="pagination mx-auto">
				<li class='page-item'><a
					href='gong_list?spage=1&sword=${sword}' class='page-link'>맨 앞</a></li>
				<c:choose>
					<c:when test="${bsu ==1 }">
						<li class='page-item'><a
							href='gong_list?spage=1&sword=${sword}' class='page-link'>이 전</a></li>
					</c:when>
					<c:otherwise>
						<li class='page-item'><a
							href='gong_list?spage=${bsu*5-5}&sword=${sword}'
							class='page-link'>이 전</a></li>
					</c:otherwise>
				</c:choose>

				<c:choose>
					<c:when test="${bsu*5 >su }">
						<c:set var="end" value="${su}" />
					</c:when>
					<c:otherwise>
						<c:set var="end" value="${bsu*5}" />
					</c:otherwise>
				</c:choose>

				<c:forEach begin="${bsu*5-4}" end="${end}" var="x">
					<c:choose>
						<c:when test="${x ==param.spage }">
							<li class='page-item active'><a
								href='gong_list?spage=${x}&sword=${sword}' class='page-link'>${x}</a></li>
						</c:when>
						<c:otherwise>
							<li class='page-item'><a
								href='gong_list?spage=${x}&sword=${sword}' class='page-link'>${x}</a></li>
						</c:otherwise>
					</c:choose>
				</c:forEach>

				<c:choose>
					<c:when test="${bsu*5 >=su }">
						<li class='page-item'><a
							href='gong_list?spage=${su}&sword=${sword}' class='page-link'>다
								음</a></li>
					</c:when>
					<c:otherwise>
						<li class='page-item'><a
							href='gong_list?spage=${bsu*5+1}&sword=${sword}'
							class='page-link'>다 음</a></li>
					</c:otherwise>
				</c:choose>
				<li class='page-item'><a
					href='gong_list?spage=${su}&sword=${sword}' class='page-link'>맨
						뒤</a></li>
			</ul>
		</div>
```
<br>
페이징 처리,검색어  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/a061111855e74a0bae2e71b090bdff7e" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>
###  게시판 - CRUD
1. 새글쓰기: 새글쓰기 같은 경우 새글을 쓰는 양식과 새글 작성 후 첫 번째 페이지로 돌아가 새 글을 보여주는 방식으로 구성하였다. 하나의 기능에 2가지 Controll이 필요하므로 같은 요청이나 Get 과 Post로서 구별하였다.
2. 자세한 내용보기: 자세한 내용보기같은 경우 게시판 Main에서 클릭했을 때 보여지는 정보를 처리하는 곳 이다. 게시판에 조회수가 있으므로 클릭했을경우 조회수를 늘리는 것을 생각하여야 한다.
3. 삭제하기: 해당 게시글을 삭제하는 기능이다. 삭제 후 삭제하기를 한 검색어와 페이지를 고려하여 삭제를 요청한 Page로 돌아가야 한다.
4. 갱신: 갱신 같은 경우 갱신을 위한 양식(갱신을 원하는 글의 정보는 담고 있는 형식)과 갱신 후 갱신을 요청한 페이지로 돌아가 갱신한 글을 보여주는 방식으로 구성하였다. 하나의 기능에 2가지 Controll이 필요하므로 같은 요청이나 Get 과 Post로서 구별하였다.


```java
	//새글 쓰기 양식
	@RequestMapping(value="gong_write",method = RequestMethod.GET)
	private String gong_write() {
		return "gong_write";
	}
	
	//새 글 쓰기
	@RequestMapping(value="gong_write",method = RequestMethod.POST)
	private void process_register(HttpServletResponse response, @RequestParam("subject") String subject,
			@RequestParam("date") String date, @RequestParam("content") String content) {
		int x = inter.maxNum();
		gongBean bean = new gongBean();
		bean.setBdate(date);
		bean.setTitle(subject);
		bean.setCon(content);
		bean.setNum(x + 1);
		inter.register(bean);
		try {
			response.sendRedirect("gong_list?spage=1");
		} catch (IOException e) {
			System.out.println("insert Error");
		}
	}

	//자세한 내용 보기
	@RequestMapping("gong_detail")
	private ModelAndView processDetail(@RequestParam("num") int num, @RequestParam("spage") int spage,
			HttpServletRequest request) {
		String sword = "";
		if (request.getParameter("sword") == null)
			sword = "All";
		else
			sword = request.getParameter("sword");

		inter.updateNum(num);
		gongDto dto = inter.detail(num);
		ModelAndView view = new ModelAndView();
		view.setViewName("gong_detail");
		view.addObject("dto", dto);
		view.addObject("spage", spage);
		view.addObject("sword", sword);
		return view;
	}

	//삭제하기
	@RequestMapping("gong_delete")
	private void processDelete(HttpServletResponse response, @RequestParam("num") String num, @RequestParam("spage") String spage, HttpServletRequest request) {
		String sword = "";
		if (request.getParameter("sword") == null)
			sword = "All";
		else
			sword = request.getParameter("sword");
		
		String word = URLEncoder.encode(sword);
		
		int x = Integer.parseInt(num);
		inter.delete(x);
		try {
			response.sendRedirect("gong_list?spage=" + spage + "&sword=" + word);
		} catch (IOException e) {
			System.out.println("Delete Error" + e);
		}
	}

	//업데이트 양식으로 가기
	@RequestMapping(value = "gong_update", method = RequestMethod.GET)
	private ModelAndView processUpdate(@RequestParam("num") int num, @RequestParam("spage") int spage,
			HttpServletRequest request) {
		String sword = "";
		if (request.getParameter("sword") == null)
			sword = "All";
		else
			sword = request.getParameter("sword");
		inter.updateNum(num);
		gongDto dto = inter.detail(num);
		ModelAndView view = new ModelAndView();
		view.setViewName("gong_update_form");
		view.addObject("dto", dto);
		view.addObject("spage", spage);
		view.addObject("sword", sword);
		return view;
	}
	
	//업데이트
	@RequestMapping(value = "gong_update", method = RequestMethod.POST)
	private void processUpdate(HttpServletResponse response, 
			HttpServletRequest request,
			@RequestParam("num") int num, 
			@RequestParam("spage") int spage,
			@RequestParam("subject") String subject,
			@RequestParam("date") String date, 
			@RequestParam("content") String content) {
		
		String sword = "";
		if (request.getParameter("sword") == null)
			sword = "All";
		else
			sword = request.getParameter("sword");
		
		String word = URLEncoder.encode(sword);
	
		gongBean bean = new gongBean();
		bean.setBdate(date);
		bean.setNum(num);
		bean.setCon(content);
		bean.setTitle(subject);
		
		inter.updateNum(num);
		inter.updateForm(bean);
		
		try {
			response.sendRedirect("gong_list?spage=" + spage + "&sword=" + word);
		} catch (IOException e) {
			System.out.println("Delete Error" + e);
		}
	}
```

<br>
개시글 추가, 수정, 삭제를 하는 동영상  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/bc9588752fd94ffda64c9d2065d5e102" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

<hr>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
---
layout: post
title:  "Spring-Project-렌트카"
date:   2019-06-27 08:00:00 +0700
categories: [Project]
---

###  렌트카-DB 구성
렌트카 구성을 위한 DB는 2개로 나누었다.  
 - 차량 등록을 위한 DB


<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Rent1.PNG" height="100%" width="100%" /></div><br>

 - 차량 예약을 위한 DB


<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Rent2.PNG" height="100%" width="100%" /></div><br>

###  렌트카-Pgae 구성
Main Page에서 Session의 id값을 비교하여 id가 admin 즉, 관리자 이면 렌트카 등록, 렌트카 모든 예약 확인이 가능하다.  
일반 유저일 경우 렌트카 확인과 렌트카 예약이 가능하다.  
Session의 id값을 비교하여 관리자를 알아내는 것은 아래 코드로 구현을 하였다.  

```jsp
<li class="nav-item dropdown">
<a class="nav-link dropdown-toggle" href="#" id="dropdown04"
	data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
	렌터카/여행자 보험</a>
	<div class="dropdown-menu" aria-labelledby="dropdown04">
		<a class="dropdown-item" href="gomycarpage?id=${id}">렌트카</a> 
			<c:if test="${id eq 'admin' }">
				<a class="dropdown-item" href="admincar">
				렌트카 모든 예약 확인</a> 
			</c:if>
				<a class="dropdown-item" href="#">여행자 보험 </a>
	</div>
</li>
```
일반 User 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Rent3.PNG" height="250" width="600" /></div><br>
<br>
Admin 화면  
<div><img src="https://raw.githubusercontent.com/wjddyd66/wjddyd66.github.io/master/static/img/Spring/Rent4.PNG" height="250" width="600" /></div><br>
<br>

###  렌트카-차량 등록
렌트카 차량 등록을 위해 정보를 입력받는 창을 만들어 입력을 받는 방식으로 하였다.  
중요한 점은 차량인 경우 사용자에게 실제 사진을 보여주는 것이 맞다고 판단되어 실제 사진은 Upload하는 것을 목표로 하였다.  
실제 File 을 Upload하기 위한 코드이다.  

pom.xml- dependency 추가
```xml
		<!-- fild upload -->
		<dependency>
			<groupId>commons-fileupload</groupId>
			<artifactId>commons-fileupload</artifactId>
			<version>1.4</version>
		</dependency>
```
<br>
실제 File을 을 위한 DTO 선언하는 과정이다.  
MultipartFile 인터페이스는 스프링에서 업로드 한 파일을 표현할 때 사용되는 인터페이스로서, MultipartFile 인터페이스를 이용해서 업로드 한 파일의 이름, 실제 데이터, 파일 크기 등을 구할 수 있다.  
<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">
<table class="table">
	<tbody>
	<tr>
		<td>Iterator<
		String
		> getFileNames()</td><td>업로드 된 파일들의 이름 목록을 제공하는 Iterator를 구한다.</td>
	</tr>
	<tr>
		<td>MultipartFile getfile(String name)</td><td>파라미터 이름이 name이 업로드 파일 정보를 구한다.</td>
	</tr>
		<tr>
		<td>List<
		MultipartFile
		> getFiles(String name)</td><td>파라미터 이름이 name인 업로드 파일 정보 목록을 구한다.</td>
	</tr>
		<tr>
		<td>Map<
		String, MultipartFile
		> getFileMap()</td><td>파라미터 이름을 키로 파라미터에 해당하는 파일 정보를 값으로 하는 Map을 구한다.</td>
	</tr>
	</tbody>
</table>
<br>

```java
package pack.rent.model;

import org.springframework.web.multipart.MultipartFile;

public class UploadFile { //fileDto
	private MultipartFile file;

	public MultipartFile getFile() {
		return file;
	}

	public void setFile(MultipartFile file) {
		this.file = file;
	}
	
}
```
<br>
실제 File의 정보와 File을 가져와서 Upload하는 Controller이다.  
```java
@Controller
public class UploadController {

	@Autowired
	@Qualifier("fileValidater")
	private FileValidater fileValidater;
	
	@Autowired
	@Qualifier("rentImpl")
	private RentInter inter;
	
	@RequestMapping(value= "imageinsert", method = RequestMethod.POST)
	public String fileUploaded(@ModelAttribute("uploadFile") UploadFile uploadFile, BindingResult result,
			@RequestParam("c_name") String c_name,
			@RequestParam("c_jong") String c_jong,
			@RequestParam("c_bun") String c_bun,
			@RequestParam("c_color") String c_color,
			@RequestParam("c_price") String c_price,
			@RequestParam("c_place") String c_place) {
		InputStream inputStream = null;
		OutputStream outputStream = null;
		MultipartFile file = uploadFile.getFile();
		fileValidater.validate(uploadFile, result);
	
		String fileName = file.getOriginalFilename();
	if (result.hasErrors()) {
		
		return "redirect:/admincar";
	}
	else {
		try {
			inputStream=file.getInputStream();
			
			File newFile = new File("C:/Users/KITCOOP/Desktop/456/BomAir_ver_Final/src/main/webapp/resources/images"+fileName);
			
			if(!newFile.exists()) {
				newFile.createNewFile();
			}
			
			outputStream = new FileOutputStream(newFile);
			int read=0;
			byte[] bytes = new byte[1024];
			
			while((read = inputStream.read(bytes))!=-1) {
				outputStream.write(bytes,0,read);
			}
			
		} catch (Exception e) {
			System.out.println("fileUploaded Error"+e);
		}finally {
			try {
				inputStream.close();
				outputStream.close();
			} catch (Exception e2) {
				System.out.println("fileUploaded Error2"+e2);
			}
		}
	}
	HashMap<String,Object> map = new HashMap<String,Object>();
	map.put("c_name",c_name);
	map.put("c_jong",c_jong);
	map.put("c_bun",c_bun);
	map.put("c_color",c_color);
	map.put("c_price",c_price);
	map.put("c_place",c_place);
	map.put("c_image",fileName);
	boolean b = inter.InsertCar(map);
	if(b) {
		return "redirect:/admincar";
	}
	return "error";
	}
}
```
<br>
실제 차량 등록  
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/aea522551a384616b5659061b74d8dd1" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>

###  렌트카-차량 삭제
실제 등록되어있는 차량을 삭제할때 실제 Upload한 File에 접근하여 삭제할 수 있게 구성하였다.  


```java
	@RequestMapping("deleteCar")
	public String deleteCar(@RequestParam("no") String no, @RequestParam("name") String name) {
		boolean b = inter.DeleteCar(no);
		boolean b2 = inter.deleteCarRent(no);
		File file = new File("C:/Users/KITCOOP/Desktop/456/BomAir_ver_Final/src/main/webapp/resources/images/"+name);
		if( file.exists() ){
            if(file.delete()){
                System.out.println("파일삭제 성공");
            }else{
                System.out.println("파일삭제 실패");
            }
        }else{
            System.out.println("파일이 존재하지 않습니다.");
        }
		
		if(b || b2) {
			return "redirect:/ListPopup";
		}
		return "error";
	}
```

<br>
실제 등록된 차량을 삭제시 사진도 같이 삭제   
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/b5a117d077194d56937007369b5cf9b9" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<br>
###  렌트카-차량 랜트
실제 등록된 차량이 있을 경우 사용자가 차량을 랜트 할 수 있다.  
랜트카의 경우 사용자가 반납을 기간내에 하거나 늦어일 수 있다는 많은 예외처리가 있다고 생각하였다.  
따라서 관리자가 직접 랜트확인을 하고 반납을 완료하는 형식으로 구성하였다.  
```java
//Rent
	@RequestMapping(value="rent", method=RequestMethod.POST)
	public ModelAndView goRentPage(@RequestParam("aa") String RentPlaceTd,
			@RequestParam("bb") String RentDateChoose,
			@RequestParam("cc") String ReturnDateChoose) {
		System.out.println(RentDateChoose);
		HashMap<String,Object> map = new HashMap<String,Object>();
		map.put("RentPlaceTd",RentPlaceTd);
		map.put("RentDateChoose",RentDateChoose);
		map.put("ReturnDateChoose",ReturnDateChoose);
		ModelAndView view=new ModelAndView("rent");
		List<RentDto> cLists = inter.RentDataAll(map);
		view.addObject("data", cLists);
		view.addObject("RentDateChoose", RentDateChoose);
		view.addObject("ReturnDateChoose", ReturnDateChoose);
		return view;
	}
	
	@RequestMapping("rentReservation")	
	public String submitRentCar(@RequestParam("RentDateChoose") String RentDateChoose,
			@RequestParam("ReturnDateChoose") String ReturnDateChoose,
			@RequestParam("rentNum") String rentNum,
			@RequestParam("rentId") String rentId,
			HttpServletRequest request, HttpServletResponse response) {
		System.out.println(RentDateChoose + " " + ReturnDateChoose + " " + rentNum);
		HashMap<String,Object> map = new HashMap<String,Object>();
		map.put("r_no",rentNum);
		map.put("c_daeil",RentDateChoose);
		map.put("c_banil",ReturnDateChoose);
		map.put("g_id",rentId);
		boolean b = inter.submitRent(map);
		if(b) {
			return "redirect:/admincar";			
		}
		return "error";
	}

//Rent 정보 확인
	@RequestMapping("gomycarpage")
	public ModelAndView gopage(@RequestParam("id") String rentId, HttpServletRequest request, HttpServletResponse response) {
		List<RentDto> myList = inter.MyRentPage(rentId);
		ModelAndView myView=new ModelAndView("mycarpage");
		myView.addObject("data",myList);	
		return myView;
	}
	
//관리자 반납 확인
	@RequestMapping("admincar")
	public ModelAndView adminCar() {
		List<RentDto> myList = inter.AdminCar();
		ModelAndView myView=new ModelAndView("mycarpage");
		myView.addObject("data",myList);
		return myView;
	}
	
	@RequestMapping("deleteRent")
	public String deleteReservation(@RequestParam("carno") String carno,
			@RequestParam("dail") String dail,
			@RequestParam("banil") String banil,
			@RequestParam("id") String id) {
		HashMap<String,Object> map = new HashMap<String,Object>();
		map.put("r_no",carno);
		map.put("c_daeil",dail);
		map.put("c_banil",banil);
		map.put("g_id",id);
		boolean b = inter.DeleteRent(map);
		if(b) {
			return "redirect:/admincar";
		}
		return "redirect:/admincar";
	}
```
아래 동영상은 
1. 사용자가 랜트 후 랜트 확인
2. 관리자가 반납 승인
3. 반납된 사용자의 랜트 정보 확인

순으로 구성되어있다.
<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe src="https://www.loom.com/embed/b82de54f33954ff794cbd341873f7768" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
<hr>
내용참조:<a href="https://devbox.tistory.com/entry/Spring-%ED%8C%8C%EC%9D%BC%EC%97%85%EB%A1%9C%EB%93%9C-%EC%B2%98%EB%A6%AC">devbox 블로그</a>
참조:<a href="https://github.com/wjddyd66/Project/tree/master/BomAir_ver_Final">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
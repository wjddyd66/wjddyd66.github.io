---
layout: post
title:  "DataAnalysis-카이제곱 검정"
date:   2019-07-29 11:30:00 +0700
categories: [DataAnalysis]
---

###  카이제곱 검정방법
카이제곱 데스트는 그룹간에 차이가 있는지 여부(= 그룹끼지 독립이 아닌지의 여부)에 대해 Chisquare 분포를 사용해 가설검정을 하는 방법이다. 그룹간에 차이가 있는지 없는지의 여부라는 의미는 그룹간의 비율차이가 있는지의 여부라는 의미이다.  

<span style ="color: red">**독립변수: 범주형, 종속변수: 범주형**</span><br>

카이제곱의 검정 방법은 목적에 따라서 3가지로 크게 나눌수 있다.  

1. 독립성 검정: 두 변수는 서로 연관성이 있는가 없는가?
2. 적합성 검정: 실제 표본이 내가 생각하는 분포와 같은가 다른가?
3. 동일성 검정: 두 집단의 분포가 동일한가? 다른 분포인가? 

카이제곱이 종류로는 크게 **일원 카이제곱 검정**, **이원 카이제곱 검정**이 존재하게 된다.  
**일원 카이제곱 검정**은 하나의 범주를 대상으로 한다. -> 적합성 검정  
**이원 카이제곱 검정**은 두개 이상의 범주 대상으로 검정 한다. -> 독립성, 동일성 검정  
참조: <a href="https://m.blog.naver.com/PostView.nhn?blogId=leerider&logNo=100189714605&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F">leerider 블로그</a>  
참조:<a href="https://wjddyd66.github.io/r/2019/06/17/Chisquare.html">카이제곱 자세한 내용</a>  
###  적합성 검사
적합성 검사 이므로 1원 카이제곱을 사용한다.  

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>주사위의 나올 확률을 모두 같다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>주사위의 나올 확률을 다르다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.014) < 
		0.05(95% 신뢰확률에서의 유의수준) => 대립가설 채택  </td>
	</tr>
	</tbody>
</table>

<br>
```python
data = [4, 6, 17, 16, 8, 9]
print(sp.stats.chisquare(data))
```
<br>
```code
Power_divergenceResult(statistic=14.200000000000001, pvalue=0.014387678176921308)
```

###  독립성 검사
동일 집단의 두 변인을 대상으로 관련성이 있는지를 판단한다.  
데이터 불러오기  
```python
data = pd.read_csv("https://raw.githubusercontent.com/wjddyd66/R/master/Data/smoke.csv")
print(data.head(3))
```
<br>
```code
   education  smoking
0          1        1
1          1        1
2          1        1
```
<br>
학력 수준별 인원수를 normalization을 한다.  
**normalization=True**로서 비율로서 표현할 수 있다.  
```python
#학력 수준별 흡연 인원수
ctab = pd.crosstab(index=data["education"], columns=data["smoking"],\
                   normalize=True)
#normalize=True => 비율로서 표현한다.
ctab.index = ["대졸", "고졸", "중졸"]
ctab.columns = ["과흡연", "보통", "비흡연"]
print(ctab)

result = stats.chi2_contingency(ctab)
print(result)
```
<br>
```code
         과흡연        보통       비흡연
대졸  0.143662  0.259155  0.191549
고졸  0.061972  0.059155  0.025352
중졸  0.121127  0.078873  0.059155
(0.05327018518268719, 0.9996515220162085, 4, array([[0.19421543, 0.23607221, 0.16407856],
       [0.04786352, 0.05817893, 0.04043642],
       [0.08468161, 0.10293196, 0.07154136]]))
```
<br>

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>학력 수준과 흡연은 관계가 없다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>학력 수준과 흡연은 관계가 있다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.99) > 
		0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택  </td>
	</tr>
	</tbody>
</table>

<br>
```python
#귀무가설: 학력 수준과 흡연은 관계가 없다.
#대립가설: 학력 수준과 흡연은 관계가 있다.
chi2, p, dof, expected = stats.chi2_contingency(ctab)
msg = "chi2:{}, p-value:{}, df:{}"
print(msg.format(chi2, p, dof))
print(expected)
#p-value(0.99) > 0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택
```
<br>
```code
chi2:0.05327018518268719, p-value:0.9996515220162085, df:4
[[0.19421543 0.23607221 0.16407856]
 [0.04786352 0.05817893 0.04043642]
 [0.08468161 0.10293196 0.07154136]]
```

###  동질성 검사
두 집간의 분포가 동일한지를 검증하는 방법이다.  

<link rel = "stylesheet" href ="/static/css/bootstrap.min.css">

<table class="table">
	<tbody>
	<tr>
		<td>귀무가설</td><td>교육방법에 따른 교육생들의 만족도 차이가 없다.</td>
	</tr>

	<tr>
		<td>대립가설</td><td>교육방법에 따른 교육생들의 만족도 차이가 있다.</td>
	</tr>
	
	<tr>
		<td>결과</td><td>p-value(0.58) > 
		0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택  </td>
	</tr>
	</tbody>
</table>

<br>
```python
#동질성 검정
#귀무가설: 교육방법에 따른 교육생들의 만족도 차이가 없다.
#대립가설: 교육방법에 따른 교육생들의 만족도 차이가 있다.
data = pd.read_csv("https://raw.githubusercontent.com/wjddyd66/R/master/Data/survey_method.csv")
print(data.head(5))

ctab = pd.crosstab(index=data["method"], columns=data["survey"])
print(ctab)
chi2, p, dof, expected = stats.chi2_contingency(ctab)
msg = "chi2:{}, p-value:{}, df:{}"
print(msg.format(chi2, p, dof))
#p-value(0.58) > 0.05(95% 신뢰확률에서의 유의수준) => 귀무가설 채택
```
<br>
```code
   no  method  survey
0   1       1       1
1   2       2       2
2   3       3       3
3   4       1       4
4   5       2       5
survey  1   2   3   4  5
method                  
1       5   8  15  16  6
2       8  14  11  11  6
3       8   7  11  15  9
chi2:6.544667820529891, p-value:0.5864574374550608, df:8
```


<hr>
참조: <a href="https://github.com/wjddyd66/DataAnalysis/blob/master/Chisquare.ipynb">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.


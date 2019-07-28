---
layout: post
title:  "R-시각화,그룹화"
date:   2019-06-16 11:30:20 +0700
categories: [R]
---

###  데이터 시각화
데이터 시각화는 데이터를 그림이나 그래프를 통해 시각적으로 표현하는 모든 과정 이다. 시각화는 단순히 멋있게 만들어 주는 것을 떠나 데이터를 쉽게 이해할 수 있도록 도와준다.  
1. 이산변수: 막대, 점, 원형 차트를 주로 사용
2. 연속변수: 상자박스, 히스토그램, 산점도차트를 주로 사용
<br>

```R
# 이산변수:막대,점,원형차트를 주로 사용
# 연속변수:상자박스,히스토그램,산점도차트를 주로 사용

stu<-read.csv("C:/git/R/Data/ex_studentlist.csv",fileEncoding = "UTF-8")
head(stu)
names(stu)

barplot(stu$grade,ylim=c(0,5),main="제목",col=rainbow(3))
plot(stu$grade,ylim=c(0,5),main="제목",col=rainbow(3)) #산점도
barplot(stu$grade,ylim=c(0,5),main="제목",col=c(1,2,3))
?barplot

par(mfrow=c(1,2)) #한 화면에 여러개의 결과를 출력할 때 사용
barplot(stu$grade,ylim=c(0,5),col=rainbow(3))
title(main="1열")

barplot(stu[,4],ylim=c(0,5),col=rainbow(3),space=2)
title(main="2열")

par(mfrow=c(1,1))
dotchart(stu$grade)
dotchart(stu$grade,color=2:5,lcolor="black",pch=1:2,cex=1.2)

df<-na.omit(stu)
str(df)
pie(df$age,labels=df$age,lty=2)
title("파이차트")

#연속변수에 적합한 차트
mean(stu$height)
median(stu$height)
quantile(stu$height)

boxplot(stu$height,range=2) #아웃라이어를 체크할 때 주로 사용한다.
boxplot(stu$height,range=2,notch = T)
abline(h=178,lty=5,col="orange")

hist(stu$height)
hist(stu$height,breaks = 10)
hist(stu$height,breaks = 10,prob=T)
lines(density(stu$height))

hist(stu$height,xlab='키',main='히스토그램',xlim=c(150,200),col='yellow')
price<-runif(10,min=1,max=100)
price
plot(price)

par(mfrow=c(2,2))
plot(price,type='l',pch=5)
plot(price,type='o',pch=10)
plot(price,type='h',pch=15,col='blue')
plot(price,type='s',pch=20,col='orange',cex=1.5)

sales<-read.csv("C:/git/R/Data/sales.csv")
par(mfrow=c(1,1))
head(sales)
attach(sales)
search()
plot(Quarter,A,type="o",col="blue",ylim=c(0,2500),axes=T,ann=T)
plot(Quarter,A,type="o",col="blue",ylim=c(0,2500),axes=F,ann=F)
plot(Quarter,B,type="o",col="blue",ylim=c(0,2500),axes=T,ann=F)
detach(sales)
search()

head(iris,2)    
attributes(iris)
pairs(iris[,1:4])
pairs(iris[iris$Species=='setosa',1:4])
pairs(~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
      data=iris,pch=c(1,2,3)[iris$Species])

#Multiple plot
par(mar=c(1,1,1,1))
layout(matrix(c(3,0,2,1),2,2,byrow=T),c(2,1),c(1,2)) 
#3,0,2,1 ->그래프의 배치가 3,0,2,1순. 0은 빈자리.
#byrow=T -> 행 우선순위 
plot(iris$Sepal.Length)
hist(iris$Sepal.Width)
boxplot(iris$Petal.Length)

par(mar=c(1,1,1,1))
install.packages("scatterplot3d")
library(scatterplot3d)
class(iris$Species)
dim(iris)
levels(iris$Species)
ir.setosa=iris[iris$Species=='setosa',]
ir.versicolor=iris[iris$Species=='versicolor',]
ir.virginica=iris[iris$Species=='virginica',]
ir.setosa
irdata<-scatterplot3d(iris$Sepal.Length,iris$Petal.Length,iris$Sepal.Width,type='n')
irdata$points3d(ir.setosa$Sepal.Length,ir.setosa$Petal.Length,ir.setosa$Sepal.Width,
                bg='red',pch=21)
irdata$points3d(ir.virginica$Sepal.Length,ir.virginica$Petal.Length,ir.virginica$Sepal.Width,
                bg='blue',pch=21)
irdata$points3d(ir.versicolor$Sepal.Length,ir.versicolor$Petal.Length,ir.versicolor$Sepal.Width,
                bg='yellow',pch=21)
```
<br>

###  그룹화
Data를 내가 원하는 형태로 만들어주는 것이 가장 큰 목적이다.  가장 많이 사용하는 것은 dplyr이다.  
dplyr

1. Select(): 데이터에서 내가 원하는 변수만 선택해주는 함수

2. Filter(): 변수에서 내가 원하는 값을 골라주는 함수

3. Group_by(): 한 변수를 기준으로 그룹화 하는 것

4. Summarise(): 원하는 통계량을 계산하여 새로운 변수에 넣어준다.

5. <span style ="color: red">%>%(파이프연산자):</span> 함수의 결과를 다시 다른 함수에 넣고 그 결과를 확인 할 수 있다.

6. Arrange(): 데이터를 오름차순 정렬해주는 함수이다. <-> Desc()

7. Join(): Merge와 같이 데이터를 매칭해서 옆으로 분여준다.

8. Mutate(): 새로운 변수를 추가해주는 함수이다.

<br>

Reshape
1. Melt: Column을 Row로 바꿈.
2. Reshape: 함수를 통해 Row를 Column으로 바꿈.
3. Dcast: Melt된 Data에 함수를 적용.

<br>
```R
#그룹화
install.packages("plyr")
library(plyr)
install.packages("dplyr")
library(dplyr)
head(iris)
unique(iris$Species)
t<-tapply(iris$Sepal.Length, iris$Species, sd) #연산대상,그룹,작업
t
class(t)

a<-plyr::ddply(iris, .(Species), summarise, avg=mean(Sepal.Length)) #데이터프레임, 집단변수
a<-plyr::ddply(iris, .(Species), summarise, avg=mean(Sepal.Length),tot=sum(Sepal.Length))
a<-plyr::ddply(iris, .(Species), summarise, avg=round(mean(Sepal.Length),1),
               tot=sum(Sepal.Length))

a
class(a)

ddply(iris,.(Species),
      function(sub){
        data.frame(Sepal_l_m=mean(sub$Sepal.Length))
      })
ddply(iris,.(Species,Sepal.Width>=3.0),
      function(sub){
        data.frame(Sepal_l_m=mean(sub$Sepal.Length))
      })

stu<-read.csv("C:/git/R/Data/ex_studentlist.csv")
stu
names(stu)

#filter
filter(stu,gender=='남',grade==2) #and
filter(stu,gender=='남'|grade==2) #or

#arrange()
arrange(stu,age)
arrange(stu,desc(age))
arrange(stu,grade,age)

#select
select(stu,name,age)
select(stu,name:age)
select(stu,-(name:age))

#summarise()
is.na(stu)
table(is.na(stu))
table(is.na(stu$age))
stdf<-na.omit(stu)
stdf
summarise(stu,avgAge=mean(age,na.rm=T))
summarise(stu,sdAge=sd(age,na.rm=T))

#%>%연산자
stu%>%filter(grade==1)
stu%>%filter(grade!=1)
stu%>%filter(height>=180.0)
stu%>%filter(height>=170.0&grade==2)
stu%>%filter(height>=170.0|grade==2)

stu%>%filter(grade%in%c(1,2))

v<-stu%>%filter(grade==1)
v
mean(v$weight)

#reshape2 package: melt(),dcast()
install.packages("reshape2")
library(reshape2)

exdf<-data.frame(id=c('a','b','c','a','b','c'),type=c(1,2,3,1,1,1),age=c(20,25,25,20,25,NA))
exdf
dcast(exdf,id~type,sum,na.tm=T) #id:행, type:열로 변경

no<-c(1,1,2,2)
day<-c(1,2,1,2)
a1<-c(40,30,50,25)
a2<-c(70,55,80,55)
df<-data.frame(no,day,a1,a2)
df

m_data<-melt(df,id=c('no','day'))
m_data

dcast(m_data,no+day~variable) #수식(행변수~열변수)
#melt로 풀어놓은 데이터를 원상태로 복귀시킨다.

dcast(m_data,no+variable~day)
dcast(m_data,no~variable+day)
dcast(m_data,no~variable,mean)
dcast(m_data,day~variable,mean)
dcast(m_data,no~day,sum)

#파일로 연습
pay_data<-read.csv("C:/git/R/Data/pay_data.csv",fileEncoding = "UTF-8")
head(pay_data)
nrow(pay_data)
table(pay_data$product_type)      

#고객별 상품유형에 따른 구매금액 출력
product_price<-dcast(pay_data,user_id~product_type,sum,na.rm=T)
head(product_price)

#고객별 지불유형에 따른 출력
product_price<-dcast(pay_data,user_id~pay_method,length)
head(product_price)
```

<hr>
참조: <a href="https://github.com/wjddyd66/R/tree/master/Visualization%26Grouping">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
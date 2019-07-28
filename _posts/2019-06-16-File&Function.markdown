---
layout: post
title:  "R-File & Function"
date:   2019-06-16 10:30:20 +0700
categories: [R]
---

###  File 읽기
Java와 마찬가지로 R에서의 File에 대한 접근에 관하여 작성한다. 
File 읽기: 읽기의 경우 크게 3가지로 나누었다.
1. 키보드 입력 받기: scan()
2. File 읽기: txt파일, Excel 파일, CSV파일: read.table(), read.xlxs(), read.csv()
3. Web에서 File읽기: Html, Xml, Json  
<br>

```R
#키보드 입력
n<-scan()
sum(1:n)
n

#R에서 파일 읽기
getwd()
list.dirs()
list.files()

student<-read.table('C:/git/R/Data/student.txt')
student

student1<-read.table('C:/git/R/Data/student1.txt',header = T,fileEncoding = "UTF-8")
student1

student2<-read.table('C:/git/R/Data/student2.txt',header = T,sep=';',skip=2,fileEncoding = "UTF-8")
student2

student4<-read.csv(file = 'C:/git/R/Data/student4.txt',sep = ',',header = T,fileEncoding = "UTF-8",na.strings = '-')
student4

#R에서 excel 읽기-read.xlsx
install.packages('xlsx')
library(xlsx)

studex<-read.xlsx('C:/git/R/Data/studentx.xlsx',sheetIndex = 1,encoding = "UTF-8")
studex

#R에서 csv읽기-read.csv
webdata<-read.csv('http://www.kma.go.kr/XML/weather/sfc_web_map.xml',header = T,encoding = 'UTF-8')
head(webdata)

#웹 스크래핑핑
#XML Data
install.packages('XML')
install.packages('httr')
library(XML)
library(httr)

url<-'https://www.melon.com/song/popup/lyricPrint.htm?songId=31754579'
source<-htmlParse(rawToChar(GET(url)$content))
source

lyrics<-xpathSApply(source,"//div[@class='box_lyric_text']",xmlValue)
lyrics

lyrics<-gsub("[\r\n\t]","",lyrics)
lyrics

#Json Data
install.packages("jsonlite")
library(jsonlite)
install.packages("httr")
library(httr)

df_repos <- fromJSON("https://api.github.com/users/hadley/repos")
str(df_repos)
doc[[1]]$owner$login

#rvest: Html File 가져오기
install.packages("rvest")
library(rvest)

?rvest

url<-"https://media.daum.net/series/"
h_daum<-read_html(url)
h_daum

h_daum%>%html_node(".item_series a")
h_daum%>%html_nodes(".item_series a")
daum<-h_daum%>%html_nodes(".item_series a")%>%html_text()
is(daum)

li<-strsplit(daum,",")
li
li[1]
li[2]
```
<br>

###  File 쓰기
File 쓰기의 경우 2가지로 나누었다.
1. print(),cat()을 통한 Console에 출력
2. sink()를 통한 File에 출력: sink란 콘솔의 출력을 그 파일로 Direction하는 것
<br>

```R
#출력
print('출력')
cat('출력')

#sink() - 콘솔의 출력을 그 파일로 디라이렉션됨
sink('output/savetest.txt') #앞으로의 작업을 파일로 저장하겠다는 선언
kbs<-9
kbs
mbc<-11
mbc
student1<-read.table('C:/git/R/Data/student1.txt',header = T,fileEncoding = "UTF-8")
student1
sink() #sink 해제

name<-c('관우','장비','유비')
age<-c(35,33,31)
gender<-c('m','m','f')
myframe<-data.frame(name,age,gender)
myframe

write.table(myframe,'C:/git/R/R_DataRead/my1.txt')
write.table(myframe,'C:/git/R/R_DataRead/my1.txt',fileEncoding = 'UTF-8')
write.table(myframe,'C:/git/R/R_DataRead/my2.txt',row.names = T)
write.table(myframe,'C:/git/R/R_DataRead/my3.txt',row.names = T)
write.table(myframe,'C:/git/R/R_DataRead/my4.txt',row.names = F,quote = F)
cat(dir('output'),sep = "\n")

write.csv(myframe,'C:/git/R/R_DataRead/my5.csv')
read.csv('C:/git/R/R_DataRead/my5.csv')

write.xlsx(myframe,'C:/git/R/R_DataRead/my6.xlsx')
read.xlsx('C:/git/R/R_DataRead/my6.xlsx',sheetIndex = 1,encoding = 'UTF-8')

#R에서 사용한 변수를 저장한 뒤 가져오기
x<-1:5
y<-6:10
save(x,y,file='C:/git/R/R_DataRead/xy.RData')
rm(list=ls())
ls()
load('C:/git/R/R_DataRead/xy.RData')
x
y
```
<br>
###  R에서의 함수
R에서의 함수도 Java와 같이 2종류로 분류 할 수 있다.  
1. 내장함수: 기존에 정의되어있는 함수
2. 사용자 정의 함수: 기존의 함수가 아닌 사용자가 원하는 대로 만든 함수

```R
#함수-내장함수,사용자정의함수

#내장함수 - 기존에 정의되어있는 함수 사용
seq(0,5,by=1,5)
rnorm(10,mean=0,sd=1)
hist(rnorm(10000,mean=0,sd=1))

#runif-난수발생
runif(10) #0~1 사이
runif(10000,min=0,max=100)
hist(runif(10,min=0,max=100))

sample(0:100,10)

vec<-1:10
min(vec)
max(vec)
range(vec)
median(vec)
var(vec)
sqrt(var(vec))
sd(vec)
quantile(vec)
prod(vec)
table(vec)
factorial(vec)
abs(-5)

#사용자정의함수 - 직접 정의하여 사용할 수 있는 함수
func1<-function(arg){
  print(arg)
  print('사용자정의함수')
}
func1('인자')

gugu_func<-function(dan){
  for(d in dan){
    for(i in 1:9){
      cat(d,'*',i,'=',d*i,'\n')
    }
    cat('\n')
  }
}
gugu_func(c(3:6))
gugu_func(c(2,7))
```
<br>

<hr>
참조: <a href="https://github.com/wjddyd66/R/tree/master/File%26Function">원본코드</a><br>
코드에 문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.
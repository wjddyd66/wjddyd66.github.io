---
layout: post
title:  "ApacheArrow"
date:   2019-08-22 10:00:00 +0700
categories: [others]
---

### ApacheArrow
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML"></script>
**ApacheArrow**란 메모리 내 에서 사로 다른 언어를 지원하기 위한 플랫폼이다.  
**ApacheArrow**는 또한 flat(DB) 하거나 hierarchical(Tree)데이터를 효율적으로 분석하기 위하여 구성되어 있다.  

**ApacheArrow**사용 이유  
1. 빠르고 언어에 구애받지 않는다.
2. 메모리 효율성이 높고, 빠른 속도를 보장한다.
3. Zero-copy
4. 로컬 및 원격 파일 시스템에 대한 IO Interface
5. 서로 다른 언어 간 binary compatibility 확인
6. 다른 메모리 내 데이터 구조와의 변환

아래 사진을 참고하면 **ApacheArrow**를 사용하는 이유를 알 수 있다.  
<div><img src="https://miro.medium.com/max/948/0*b3jEIVWk_m0_J9jH.png" height="100%" width="100%" /></div><br>
**ApacheArrow는 Columnar Buffer를 활용하여 IO를 줄이고 분석 처리 성능을 가속화 한다.**  


### Apache Arrow 사용 예시

#### Converting Pandas Dataframe to Apache Arrow Table

```python
df = pd.DataFrame({'one': [20, np.nan, 2.5],'two': ['january', 'february', 'march'],'three': [True, False, True]},index=list('abc'))
table = pa.Table.from_pandas(df)
```

#### Pyarrow Table to Pandas Data Frame

```python
df_new = table.to_pandas()
```

#### Read CSV

```python
from pyarrow import csv
fn = 'data.csv'
table = csv.read_csv(fn)
df = table.to_pandas()
```

#### Writing a parquet file from Apache Arrow

```python
import pyarrow.parquet as pq
pq.write_table(table, 'example.parquet')
```

#### Reading a parquet file

```python
table2 = pq.read_table('example.parquet')
table2
```

```code
pyarrow.Table
no: int64
tv: double
radio: double
newspaper: double
sales: double
```

#### Reading some columns from a parquet file

```python
table2 = pq.read_table('example.parquet', columns=['one', 'three'])
```

#### Transforming Parquet file into a Pandas DataFrame

```python
pdf = pq.read_pandas('example.parquet', columns=['two']).to_pandas()
pdf
```

#### Avoiding pandas index

```python
table = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(table, 'example_noindex.parquet')
t = pq.read_table('example_noindex.parquet')
t.to_pandas()
```

```code
	no	tv	radio	newspaper	sales
0	1	230.1	37.8	69.2	22.1
1	2	44.5	39.3	45.1	10.4
2	3	17.2	45.9	69.3	9.3

...

196	197	94.2	4.9	8.1	9.7
197	198	177.0	9.3	6.4	12.8
198	199	283.6	42.0	66.2	25.5
199	200	232.1	8.6	8.7	13.4
200 rows × 5 columns
```

#### Check metadata

```python
parquet_file = pq.ParquetFile('example.parquet')
parquet_file.metadata
```

```code
<pyarrow._parquet.FileMetaData object at 0x0000000007FEC3B8>
  created_by: parquet-cpp version 1.5.1-SNAPSHOT
  num_columns: 5
  num_rows: 200
  num_row_groups: 1
  format_version: 1.0
  serialized_size: 550
```

#### See data schema

```python
parquet_file.schema
```

```code
<pyarrow._parquet.ParquetSchema object at 0x0000000007FF3188>
no: INT64
tv: DOUBLE
radio: DOUBLE
newspaper: DOUBLE
sales: DOUBLE
```

#### In Memory
```python
writer = pa.BufferOutputStream()
writer.write(b'hello, friends')
buf = writer.getvalue()
print(buf)
print(buf.size)
```

```code
<pyarrow.lib.Buffer object at 0x0000000007E067F0>
14
```

```python
reader = pa.BufferReader(buf)
reader.seek(0)
reader.read(14)
```

```code
b'hello, friends'
```

참조: <a href="https://towardsdatascience.com/a-gentle-introduction-to-apache-arrow-with-apache-spark-and-pandas-bb19ffe0ddae">**PyArrow 사용 방법**</a>  


<br>
<br>
<hr>
참조: <a href="https://github.com/wjddyd66/others/blob/master/Project/ApacheArrowExample.ipynb">원본코드</a><br> 
문제가 있거나 궁금한 점이 있으면 wjddyd66@naver.com으로  Mail을 남겨주세요.



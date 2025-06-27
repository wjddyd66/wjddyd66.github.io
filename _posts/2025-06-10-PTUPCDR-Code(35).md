---
layout: post
title:  "Paper35. Personalized Transfer of User Preferences for Cross-domain Recommendation-Code"
date:   2025-06-10 11:00:20 +0700
categories: [Paper]
---
<script type="text/x-mathjax-config">
MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

### Setting

#### Import Library


```python
import os
import json
import gzip
import tqdm
import random
import argparse
import numpy as np
import pandas as pd

import torch
import keras
from torch.utils.data import DataLoader, TensorDataset
```

#### Argument


```python
def prepare(config_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=0)
    parser.add_argument('--task', default='1')
    parser.add_argument('--base_model', default='MF')
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--ratio', default=[0.8, 0.2])
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args(args=[])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(config_path, 'r') as f:
        config = json.load(f)
        config['base_model'] = args.base_model
        config['task'] = args.task
        config['ratio'] = args.ratio
        config['epoch'] = args.epoch
        config['lr'] = args.lr
    return args, config

config_path = 'config.json'
args, config = prepare(config_path)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
```

### Data preprocessing

#### Raw Dataset Load

해당 과정은 각 Domain별로, Dataset을 [uid, iid, rating]으로서 만드는 과정이다. 해당 Dataset을 살펴보면 아래와 같다.

#### Original Code


```python
# class DataPreprocessingMid():
#     def __init__(self,
#                  root,
#                  dealing):
#         self.root = root
#         self.dealing = dealing

#     def main(self):
#         print('Parsing ' + self.dealing + ' Mid...')
#         re = []
#         # with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json', 'rb') as f:
#         #     for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
#         #         line = json.loads(line)
#         #         re.append([line['reviewerID'], line['asin'], line['overall']])
#         with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
#             for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
#                 line = json.loads(line)
#                 re.append([line['reviewerID'], line['asin'], line['overall']])
#         re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
#         print(self.dealing + ' Mid Done.')
#         re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
#         return re
        
# if args.process_data_mid:
#     for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
#         DataPreprocessingMid(config['root'], dealing).main()
```

#### Code 펼치기


```python
book_data = pd.read_csv(config['root'] + 'mid/' + 'Books' + '.csv')
music_data = pd.read_csv(config['root'] + 'mid/' + 'CDs_and_Vinyl' + '.csv')
movie_data = pd.read_csv(config['root'] + 'mid/' + 'Movies_and_TV' + '.csv')
```


```python
print("EDA Each Dataset")
for data_name, data in zip(['Book', 'Music', 'Movie'], [book_data, music_data, movie_data]):
    print(data_name)
    print('User: {}, Item: {}, Rating (Number of Interaction): {}'.format(data['uid'].nunique(),
                                                                          data['iid'].nunique(),
                                                                          len(data)))
    print('Sparsity: {}'.format(round(len(data)/(data['uid'].nunique() * data['iid'].nunique()), 5)))
    print('-'*50)
    print('\n\n')
```

    EDA Each Dataset
    Book
    User: 603668, Item: 367982, Rating (Number of Interaction): 8898041
    Sparsity: 4e-05
    --------------------------------------------------


​    
​    
    Music
    User: 75258, Item: 64443, Rating (Number of Interaction): 1097592
    Sparsity: 0.00023
    --------------------------------------------------


​    
​    
    Movie
    User: 123960, Item: 50052, Rating (Number of Interaction): 1697533
    Sparsity: 0.00027
    --------------------------------------------------


​    
​    
​    


```python
print("EDA Pair Dataset")

print('Task1, Source: Movie, Target: Music')
print('Overlap User: {}'.format(len(list(set(movie_data['uid'].unique()) & set(music_data['uid'].unique())))))
print('-'*50)
print('\n\n')

print('Task2, Source: Book, Target: Movie')
print('Overlap User: {}'.format(len(list(set(book_data['uid'].unique()) & set(movie_data['uid'].unique())))))
print('-'*50)
print('\n\n')

print('Task3, Source: Book, Target: Music')
print('Overlap User: {}'.format(len(list(set(book_data['uid'].unique()) & set(music_data['uid'].unique())))))
```

    EDA Pair Dataset
    Task1, Source: Movie, Target: Music
    Overlap User: 18031
    --------------------------------------------------


​    
​    
    Task2, Source: Book, Target: Movie
    Overlap User: 37388
    --------------------------------------------------


​    
​    
    Task3, Source: Book, Target: Music
    Overlap User: 16738


**아래 Reproduction 진행은 빠르게 확인하기 위하여, Dataset이 상대적으로 적은 Task1으로서 Reproduction 수행 진행**

#### Cross Domain Dataset

위의 과정은 Raw Dataset을 만드는 과정이 였다. 아래 과정은 실제 Dataset을 기준으로 Cross Domain Dataset을 만드는 과정이다.

#### Original Code


```python
# class DataPreprocessingReady():
#     def __init__(self,
#                  root,
#                  src_tgt_pairs,
#                  task,
#                  ratio):
#         self.root = root
#         self.src = src_tgt_pairs[task]['src']
#         self.tgt = src_tgt_pairs[task]['tgt']
#         self.ratio = ratio

#     def read_mid(self, field):
#         path = self.root + 'mid/' + field + '.csv'
#         re = pd.read_csv(path)
#         return re

#     def mapper(self, src, tgt):
#         print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
#         print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
#         co_uid = set(src.uid) & set(tgt.uid)
#         all_uid = set(src.uid) | set(tgt.uid)
#         print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
#         uid_dict = dict(zip(all_uid, range(len(all_uid))))
#         iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
#         iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
#         src.uid = src.uid.map(uid_dict)
#         src.iid = src.iid.map(iid_dict_src)
#         tgt.uid = tgt.uid.map(uid_dict)
#         tgt.iid = tgt.iid.map(iid_dict_tgt)
#         return src, tgt

#     def get_history(self, data, uid_set):
#         pos_seq_dict = {}
#         for uid in tqdm.tqdm(uid_set):
#             pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
#             pos_seq_dict[uid] = pos
#         return pos_seq_dict

#     def split(self, src, tgt):
#         print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
#         src_users = set(src.uid.unique())
#         tgt_users = set(tgt.uid.unique())
#         co_users = src_users & tgt_users
#         test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))
#         train_src = src
#         train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
#         test = tgt[tgt['uid'].isin(test_users)]
#         pos_seq_dict = self.get_history(src, co_users)
#         train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
#         train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
#         test['pos_seq'] = test['uid'].map(pos_seq_dict)
#         return train_src, train_tgt, train_meta, test

#     def save(self, train_src, train_tgt, train_meta, test):
#         output_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
#                       '/tgt_' + self.tgt + '_src_' + self.src
#         if not os.path.exists(output_root):
#             os.makedirs(output_root)
#         print(output_root)
#         train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
#         train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
#         train_meta.to_csv(output_root +  '/train_meta.csv', sep=',', header=None, index=False)
#         test.to_csv(output_root + '/test.csv', sep=',', header=None, index=False)

#     def main(self):
#         src = self.read_mid(self.src)
#         tgt = self.read_mid(self.tgt)
#         src, tgt = self.mapper(src, tgt)
#         train_src, train_tgt, train_meta, test = self.split(src, tgt)
#         self.save(train_src, train_tgt, train_meta, test)

# if args.process_data_ready:
#     for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
#         for task in ['1', '2', '3']:
#             DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()
```

#### Code 펼치기

#### Step1. Load Raw Dataset

Task1
- src: Movie
- tgt: Music


```python
path = config['root'] + 'mid/' + 'Movies_and_TV' + '.csv'
src = pd.read_csv(path)

path = config['root'] + 'mid/' + 'CDs_and_Vinyl' + '.csv'
tgt = pd.read_csv(path)
```

#### Step2. Preprocessing

각각의 uid와 iid를 mapping하는 과정을 진행한다. **주요한 점은, user는 overlap되는 상황이고, iid는 overlap되지 않는 상황이다.  -> Code를 그대로 사용하려면, 후에 item이 overlap되는지 판단하여 코드 수정이 필요한 부분인다.**


```python
print('Before Mapping Example')
print('User id: {}, Item id: {}'.format(src.uid[0], src.iid[0]))
```

    Before Mapping Example
    User id: ADZPIG9QOCDG5, Item id: 0005019281



```python
print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))

# Overlap 되는 User
co_uid = set(src.uid) & set(tgt.uid)

# Overlap을 고려하여, 중복되지 않은 모든 User
all_uid = set(src.uid) | set(tgt.uid)
print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))

# User mapping file은 위에서 생성한 all_uid로 생성
uid_dict = dict(zip(all_uid, range(len(all_uid))))

# Item은 겹치지 않으므로, 아래와 같은 방법으로 mapping (target item id = original target item id + max(source item id))
iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))

# 위에서 생성한 mapping file을 활용하여 mapping
src.uid = src.uid.map(uid_dict)
src.iid = src.iid.map(iid_dict_src)
tgt.uid = tgt.uid.map(uid_dict)
tgt.iid = tgt.iid.map(iid_dict_tgt)
```

    Source inters: 1697533, uid: 123960, iid: 50052.
    Target inters: 1097592, uid: 75258, iid: 64443.
    All uid: 181187, Co uid: 18031.



```python
print('After Mapping Example')
print('User id: {}, Item id: {}'.format(src.uid[0], src.iid[0]))
```

    After Mapping Example
    User id: 102451, Item id: 19610


#### <font color=red>주의 사항!!!!</font>
나중에 코드를 아래와 같이 수정하여서 사용하여야 한다. 왜냐하면, 우리는 Interaction Item histroy의 set length를 맞춰주기 위하여 padding값을 임의로 추가할 것 이다.  
**해당 값은 default값으로 0으로 채울 것 이며, 따라서 source domain의 item idx=1부터 시작하여야 한다.**  
따라서, 해당 코드를 아래와 같이 수정하여 수정하여야 한다.

```python
iid_dict_src = dict(zip(set(src.iid), range(1, len(set(src.iid))+1)))
iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid))+1, len(set(src.iid))+1 + len(set(tgt.iid)))))
```

**또한, 밑에서 계속적으로 사용할 iid_all의 경우에도 +1로서 잊지말고 진행하여야 한다.**


```python
print('Source Domain Item id: {} ~ {}'.format(min(iid_dict_src.values()), max(iid_dict_src.values())))
print('Target Domain Item id: {} ~ {}'.format(min(iid_dict_tgt.values()), max(iid_dict_tgt.values())))
```

    Source Domain Item id: 0 ~ 50051
    Target Domain Item id: 50052 ~ 114494


#### Step3. Split Train & Test

Train & Test로 Split하는 과정이다. 또한 해당 과정에서 고려할 점은 **Characteristic Encoder를 위하여 Source Domain의 User의 Interacted item in source domain이 필요로 하다. (Time Sequence는 고려하지 않는다.)**


```python
print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
src_users = set(src.uid.unique())
tgt_users = set(tgt.uid.unique())
co_users = src_users & tgt_users
        
# Test User의 경우에는 Source Domain과 Target Domain에 모두 존재하는 User로서 진행 된다.
test_ratio = 0.2 # 임의로 설정
test_users = set(random.sample(co_users, round(test_ratio * len(co_users))))

'''
아래 코드에서 주요하게 봐야할 점은 2가지 이다.
1. Source Domain에서 Test User는 사용하여도 된다. -> Source Domain에서의 Interaction정보를 사용하지만, Target Domain에서의 Interaction은 사용하지 않아 치팅이 아니다.
2. Target Domain에서는 Test User는 무조건 제외되어야 한다.
'''
train_src = src
train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
test = tgt[tgt['uid'].isin(test_users)]
```

    All iid: 114495.


#### Step4. Make Meta Network Dataset

위의 과정에서는 일반적인 Ranking Model을 만들기 위하여 Dataset을 구성하는 과정과 동일하다는 것을 알 수 있다.  
하지만, Source Domain User Embedding -> Target Domain User Embedding을 진행하기 위하여 아래와 같은 과정이 필요하다.

1. Source Domain에서 User-Item Interaction set을 구해야 한다.
2. Source User Embdding -> Target User Embedding을 학습하기 위한 데이터셋을 구성하여야 한다.


```python
'''
해당 과정은 Source Domain에서 User-Item Interaction set을 구하는 과정이다.
일반적인 movielens와 동일하게 4~5의 값을 positive로서 설정하였다.
'''
def get_history(data, uid_set):
    pos_seq_dict = {}
    for uid in tqdm.tqdm(uid_set):
        pos = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
        pos_seq_dict[uid] = pos
    return pos_seq_dict

# 1번 과정
pos_seq_dict = get_history(src, co_users)

# 2번 과정
train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)

# testset도 inference를 위하여 아래와 같이 interaction set mapping
test['pos_seq'] = test['uid'].map(pos_seq_dict)
```

    100%|████████████████████████████████████| 18031/18031 [01:34<00:00, 191.29it/s]
    /var/folders/1d/07mdpnmd1dqdqssv2lr6_kjw0000gn/T/ipykernel_83376/3892097877.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      train_meta['pos_seq'] = train_meta['uid'].map(pos_seq_dict)
    /var/folders/1d/07mdpnmd1dqdqssv2lr6_kjw0000gn/T/ipykernel_83376/3892097877.py:20: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      test['pos_seq'] = test['uid'].map(pos_seq_dict)


#### Example

#### Train Source Dataset


```python
train_src.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>iid</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>102451</td>
      <td>19610</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83121</td>
      <td>19610</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>173894</td>
      <td>19610</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Train Target Dataset


```python
train_tgt.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>iid</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>97750</td>
      <td>97032</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>135504</td>
      <td>97032</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58341</td>
      <td>97032</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Meta Dataset


```python
train_meta.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>iid</th>
      <th>y</th>
      <th>pos_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>135504</td>
      <td>97032</td>
      <td>4.0</td>
      <td>[36381, 12811, 17848, 23390, 22093, 11073, 666...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>58341</td>
      <td>97032</td>
      <td>5.0</td>
      <td>[36381, 12699, 32488, 11444, 50017, 8363, 1996...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>111177</td>
      <td>97032</td>
      <td>5.0</td>
      <td>[36381, 27589, 32976, 20387, 38915, 30418, 102...</td>
    </tr>
  </tbody>
</table>
</div>



#### Test Dataset


```python
test
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>uid</th>
      <th>iid</th>
      <th>y</th>
      <th>pos_seq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>44038</td>
      <td>97032</td>
      <td>5.0</td>
      <td>[36381, 37403, 10320, 47285, 28702, 22112, 352...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>167378</td>
      <td>66605</td>
      <td>5.0</td>
      <td>[17212, 20845, 35039, 7416, 47362, 10478, 2249...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>39472</td>
      <td>66605</td>
      <td>3.0</td>
      <td>[35717, 23226, 32074, 34414, 16479, 14579]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>77758</td>
      <td>69705</td>
      <td>3.0</td>
      <td>[23545, 28420, 20675, 14413, 15871, 5110, 4243...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>56652</td>
      <td>69705</td>
      <td>5.0</td>
      <td>[6096, 15248, 19270, 16479, 2473, 1678, 33345,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1097541</th>
      <td>54145</td>
      <td>107741</td>
      <td>1.0</td>
      <td>[6096, 6435, 45924, 26097, 35439, 44765, 29171...</td>
    </tr>
    <tr>
      <th>1097550</th>
      <td>136000</td>
      <td>107741</td>
      <td>2.0</td>
      <td>[32488, 6096, 30978, 20482, 10149, 29462, 2886...</td>
    </tr>
    <tr>
      <th>1097553</th>
      <td>167229</td>
      <td>107741</td>
      <td>4.0</td>
      <td>[7399, 37679, 42072, 37595, 15429, 44627]</td>
    </tr>
    <tr>
      <th>1097562</th>
      <td>92170</td>
      <td>62886</td>
      <td>5.0</td>
      <td>[27072, 18565, 47209, 8287, 7154]</td>
    </tr>
    <tr>
      <th>1097581</th>
      <td>102416</td>
      <td>62886</td>
      <td>5.0</td>
      <td>[44349, 33626, 11969, 44844]</td>
    </tr>
  </tbody>
</table>
<p>97686 rows × 4 columns</p>
</div>



### Model

실제 Model의 구성을 살펴보면 아래와 같이 3가지로 구성되어 있다.
1. MF Based
2. DNN Based
3. GMF Based
   

**Appendix:** GMF에 대해서는 해당 논문에서 다음과 같이 얘기하고 있다. GMF assigns various weights for different dimensions in the dot-product prediction function, which can be regarded as as generalization of vanilla MF. 즉, MF에 Prediction Layer를 dot-product로 수행한다는 이야기 이다.

#### Embedding Module

CDR을 구성하기 위한, 각 uid, iid를 embedding하기 위한 각 module(MF, DNN, GMF)에 대한 설명 이다.

**1. MF Based Embedding (Lookup Embedding)**  

**1.1. Embedding Step**  
uid, iid를 Embedding하기 위하여 필요한 Module이다.
- input
    - uid: User ID
    - iid: Item ID
- output: [user embedding, item embedding] (concat)
    - shape: [batch, 2(user, id), embedding dim]

주의 사항: padding값이 0으로 채워지기 때문에, iid는 iid_all+1로 지정하여야 한다.

**Future Work: 후에, User Feature를 통하여 Embedding하게 된다면, 해당 부분을 AE or PCA등으로 수정하여야 한다.**


```python
class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb
```

**1.2. Prediction Step**  
위의 Embedding을 학습하기 위하여, Prediction 과정이 필요하다. 해당 부분은 아래와 같이 dot-product로서 구현하였다.


```python
for x,y in data_src:
    break

print('Input: [uid, iid], shape: {}'.format(x.shape))
print('Input Example: {}'.format(x[0]))
print('\n\nLabel: [score], shape: {}'.format(y.shape))
print('Label Example: {}'.format(y[0]))

# MF Embedding
mf_emb_model = LookupEmbedding(uid_all, iid_all, config['emb_dim'])
emb = mf_emb_model(x)
print('\n\nEmbedding Output: [user embedding, item embedding] (concat), shape: {}'.format(emb.shape))
print('Embedding Output Example: {}'.format(emb[0]))

# MF Prediction
prediction = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
print('\n\nPrediction Output Shape: {}'.format(prediction.shape))
print('Prediction Output Example: {}'.format(prediction[0]))
```

    Input: [uid, iid], shape: torch.Size([256, 2])
    Input Example: tensor([103345,   6949])


​    
    Label: [score], shape: torch.Size([256, 1])
    Label Example: tensor([3])


​    
    Embedding Output: [user embedding, item embedding] (concat), shape: torch.Size([256, 2, 10])
    Embedding Output Example: tensor([[ 0.3977, -0.4060, -1.4346,  0.8223, -0.8921, -0.8643, -1.0019, -0.5621,
             -1.4699, -0.4680],
            [ 0.6178, -1.4081, -0.5229, -0.1725,  0.7751, -0.3450, -1.3547, -1.5736,
             -1.1587, -0.7215]], grad_fn=<SelectBackward0>)


​    
    Prediction Output Shape: torch.Size([256])
    Prediction Output Example: 5.31486177444458


**2. GMF Based Embedding**  

**2.1. Embedding Step**  
MF와 동일

**2.2. Prediction Step**  
해당 Module을 MF의 결과에 대하여 Prediction Layer를 dot-product후, Layer를 통하여 Prediction을 진행한다.


```python
class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)
```


```python
# GMF Embedding
gmf_emb_model = GMFBase(uid_all, iid_all, config['emb_dim'])
emb = gmf_emb_model.embedding(x)
print('\n\nEmbedding Output: [user embedding, item embedding] (concat), shape: {}'.format(emb.shape))
print('Embedding Output Example: {}'.format(emb[0]))

# GMF Prediction
prediction = gmf_emb_model(x)
print('\n\nPrediction Output Shape: {}'.format(prediction.shape))
print('Prediction Output Example: {}'.format(prediction[0]))
```


​    
    Embedding Output: [user embedding, item embedding] (concat), shape: torch.Size([256, 2, 10])
    Embedding Output Example: tensor([[ 0.3286,  0.0331,  0.1995,  0.1515,  0.1289, -1.0519, -0.4489, -0.0242,
             -1.7985,  1.1379],
            [ 0.1754,  0.0429,  0.7566,  0.6509, -1.9287,  0.9130, -1.7802, -0.0348,
             -1.6982,  1.0922]], grad_fn=<SelectBackward0>)


​    
    Prediction Output Shape: torch.Size([256])
    Prediction Output Example: -0.7744709253311157


**3. DNN Based Embedding**  

**3.1. Embedding Step**  
uid, iid -> Embedding과정은 MF와 동일하지만, **Feed-Forward Network를 하나 더 거친다는 것이 다른 부분이다.**

**2.2. Prediction Step**  
MF와 동일


```python
class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x
```


```python
# DNN Embedding
dnn_emb_model = DNNBase(uid_all, iid_all, config['emb_dim'])
emb = gmf_emb_model.embedding(x)
print('\n\nEmbedding Output: [user embedding, item embedding] (concat), shape: {}'.format(emb.shape))
print('Embedding Output Example: {}'.format(emb[0]))

# DNN Prediction
prediction = dnn_emb_model(x)
print('\n\nPrediction Output Shape: {}'.format(prediction.shape))
print('Prediction Output Example: {}'.format(prediction[0]))
```


​    
    Embedding Output: [user embedding, item embedding] (concat), shape: torch.Size([256, 2, 10])
    Embedding Output Example: tensor([[ 0.3286,  0.0331,  0.1995,  0.1515,  0.1289, -1.0519, -0.4489, -0.0242,
             -1.7985,  1.1379],
            [ 0.1754,  0.0429,  0.7566,  0.6509, -1.9287,  0.9130, -1.7802, -0.0348,
             -1.6982,  1.0922]], grad_fn=<SelectBackward0>)


​    
    Prediction Output Shape: torch.Size([256])
    Prediction Output Example: -0.48620307445526123


#### Mapping Function

<font color=red>해당 Module은 실제 다른 Dataset을 활용할 때는 의미 없는 Module 입니다. 해당 Simulation Dataset에 맞는 과정을 추가하기 위하여 해당 과정을 진행하였고, 실제 Domain에서는 Target Item Embedding, Target User Embedding을 생성하여 활용하여야 한다.</font>

먼저 dataset은 아래와 같이 구성되어 있다. (밑의 코드에서 mapping data에 대하여 정의 되어 있다.)
- input: source domain user id
- output: target domain user id

해당 데이터를 활용하여 embedding을 mapping하는 방식 이다. 해당 과정이 필요한 이유는 <span>$$L = \sum_{u_i \in \mathcal{U}^o} \| \hat{u}_i^t - u_i^t\|^2$$</span>을 학습하기 위해서 이다.

- <span>$$u_i^s$$</span>: Source User Embedding Prediction // <code>sself.src_model.linear(...)</code>
- <span>$$\hat{u}_i^t = f_{u_i}(u_i^s ; w_{u_i})$$</span>: Target User Embedding Prediction. <span>$$w_{u_i}$$</span> is trainable parameter // <code>elf.mapping.forward(src_emb)</code>
- <span>$$u_i^t$$</span>: Target User Embedding // <code>self.tgt_model.linear(...)</code>

해당 구현에 대해서는 아래와 같이 간단하게 구현되어 있다.

```python
# Mapping Define
self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)

# Source User Embedding -> Target User Embedding
elif stage == 'train_map':
    src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()) # Source Embedding
    src_emb = self.mapping.forward(src_emb) # Source Embedding -> Target Embedding Prediction
    tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze()) # Target Embedding
    return src_emb, tgt_emb
```

Example of Mapping Dataset


```python
for x,y in data_map:
    break
```


```python
print('Source User ID')
print(x[:3])

print('\n\nTarget User ID')
print(y[:3])
```

    Source User ID
    tensor([168284, 126675,   4648])


​    
    Target User ID
    tensor([   58,  5642, 11526])


#### Meta Network

Source Domain의 User Embedding + User-Item historical interaction으로서 Target Source Domain의 User Embedding을 만드는 과정이다. 해당 과정에 대해서는 아래와 같이 정의하였었다.

**PreDefine**  
- <span>$$u_i^d \in \mathbb{R}^k, v_i^d \in \mathbb{R}^k, \text{s.t. }d \in \{ s,t \}$$</span>: Item and User Ebmedding with dimension k
- <span>$$S_{u_i} = \{ v_{t_1}^s, v_{t_2}^s, \ldots, v_{t_n}^s\}$$</span>: Source Domain(s)에서 User(<span>$$u_i$$</span>)가 TimeStamp(<span>$$t_n$$</span>)까지 Interaction이 있었던 모든 Item Set

**Input**  
- <span>$$S_{u_i}$$</span>: <code>emb_fea</code>, Batch Size (B)x Sequence Length (S)x Embedding Dimension (E)
- <code>seq_index</code>: Item Sequence, B x S. 0=padding

논문에서 구현한 수식을 하나하나 따라가면 아래와 같다.

**Charcteristic Encoder**  

Notation
<p>$$p_{u_i} = \sum_{v_j^s \in S_{u_i}} a_j v_j^s$$</p>
<p>$$p_{u_i} \in \mathbb{R}^k$$</p>
<p>$$a_j = \frac{\text{exp}(a_{j}^{'})}{\sum_{v_j^s \in S_{u_i}} \text{exp}(a_{j}^{'})}$$</p>
<p>$$a_{j}^{'} = h(v_j ; \theta),$$</p>
<p>$$h(\cdot) \text{ is two-layer feed-forward network}$$</p>

Code 구현
- <span>$$a_{j}^{'} = h(v_j ; \theta)$$</span>: <code>event_K = self.event_K(emb_fea)</code>: B x S x 1
- <code>t = event_K - torch.unsqueeze(mask, 2) * 1e8</code>: 해당 부분은 padding인 item sequence는 제외하고 값을 계산하기 위하여, padding인 부분의 값은 -1e+8, 나머지는 Score값을 그대로 사용한다. // B x S x 1
- <span>$$a_j$$</span>: <code>att = self.event_softmax(t)</code>: Attention Score
- <span>$$p_{u_i} = \sum_{v_j^s \in S_{u_i}} a_j v_j^s$$</span>: <code>his_fea = torch.sum(att * emb_fea, 1)</code>: B x E

**Meta Network**  
- <span>$$w_{u_i} = g(p_{u_i} ; \phi) \in \mathbb{R}^{k \times k}$$</span>: <code>output = self.decoder(his_fea)</code>: B x E^2


```python
class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)
```

**Personalized Bridge**  
- <span>$$\hat{u}_i^t = f_{u_i}(u_i^s ; w_{u_i})$$</span>: <code>uid_emb = torch.bmm(uid_emb_src, mapping)</code>: B x E

Code Example


```python
src_model = DNNBase(uid_all, iid_all, emb_dim)
tgt_model = DNNBase(uid_all, iid_all, emb_dim)

meta_net = MetaNet(emb_dim, meta_dim)
```


```python
for x,y in data_meta:
    break
```


```python
iid_emb = tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1)) # B x 1 x E
uid_emb_src = src_model.linear(src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))) # B x 1 x E
ufea = src_model.embedding.iid_embedding(x[:, 2:]) # B x S x E

mapping = meta_net.forward(ufea, x[:, 2:]).view(-1, emb_dim, emb_dim) # B x E x E
uid_emb = torch.bmm(uid_emb_src, mapping) # B x 1 x E
```


```python
print('User Output Shape: {}'.format(uid_emb.shape))
```

    User Output Shape: torch.Size([128, 1, 10])


<font color=red>위의 과정에서 단순히, B x E x E 로 Decoding하는 이유를 잘 모르겠다. 그냥, 단순하게 Personalized Bridge. 즉, Target Domain으로 어떻게 mapping할지, 하나로 정한 것 같다. 딱히 의미가 있을 것 같지는 않다.ㅇ</font>

### Training

Model CDR 학습전에 알아두어야 할 사항은, 아래와 같다.
1. 각 Domain의 User, Item Embedding Vector를 생성 (PreTraining 과정)
2. Source Domain User Embedding -> Target Domain User Embedding (Meta Network)
3. Ranking Algorithm

아래 코드 Line-by-Line으로 어떻게 학습이 진행되는지 확인한다.

**Appendix: 해당 논문에서는 Embedding 방법은 크게 3가지로 구성하였습니다. 각각의 방법은 아래와 같습니다.**
1. Look Up Embedding을 활용하여 구성하였습니다.
2. GMF 방식 (Wip)
3. DNN 방식 (Wip)


#### Step1. Setting Configure


```python
use_cuda = config['use_cuda']
base_model = config['base_model'] # 여기서 위에서 지정한 3개의 Model중 어떠한 것을 쓸지 정한다.

root = config['root']
ratio = config['ratio']
task = config['task']
src = config['src_tgt_pairs'][task]['src']
tgt = config['src_tgt_pairs'][task]['tgt']
uid_all = config['src_tgt_pairs'][task]['uid']
iid_all = config['src_tgt_pairs'][task]['iid']

batchsize_src = config['src_tgt_pairs'][task]['batchsize_src']
batchsize_tgt = config['src_tgt_pairs'][task]['batchsize_tgt']
batchsize_meta = config['src_tgt_pairs'][task]['batchsize_meta']
batchsize_map = config['src_tgt_pairs'][task]['batchsize_map']
batchsize_test = config['src_tgt_pairs'][task]['batchsize_test']
batchsize_aug = batchsize_src

epoch = config['epoch']
emb_dim = config['emb_dim']
meta_dim = config['meta_dim']
num_fields = config['num_fields']
lr = config['lr']
wd = config['wd']

input_root = root + 'ready/_' + str(int(ratio[0] * 10)) + '_' + str(int(ratio[1] * 10)) + '/tgt_' + tgt + '_src_' + src
src_path = input_root + '/train_src.csv'
tgt_path = input_root + '/train_tgt.csv'
meta_path = input_root + '/train_meta.csv'
test_path = input_root + '/test.csv'

results = {'tgt_mae': 10, 'tgt_rmse': 10,
           'aug_mae': 10, 'aug_rmse': 10,
           'emcdr_mae': 10, 'emcdr_rmse': 10,
           'ptupcdr_mae': 10, 'ptupcdr_rmse': 10}
```


```python
input_root
```




    './data/ready/_8_2/tgt_CDs_and_Vinyl_src_Movies_and_TV'



#### Step2. Prepare Dataset

#### Original Code


```python
# def seq_extractor(self, x):
#         x = x.rstrip(']').lstrip('[').split(', ')
#         for i in range(len(x)):
#             try:
#                 x[i] = int(x[i])
#             except:
#                 x[i] = self.iid_all
#         return np.array(x)

#     def read_log_data(self, path, batchsize, history=False):
#         if not history:
#             cols = ['uid', 'iid', 'y']
#             x_col = ['uid', 'iid']
#             y_col = ['y']
#             data = pd.read_csv(path, header=None)
#             data.columns = cols
#             X = torch.tensor(data[x_col].values, dtype=torch.long)
#             y = torch.tensor(data[y_col].values, dtype=torch.long)
#             if self.use_cuda:
#                 X = X.cuda()
#                 y = y.cuda()
#             dataset = TensorDataset(X, y)
#             data_iter = DataLoader(dataset, batchsize, shuffle=True)
#             return data_iter
#         else:
#             data = pd.read_csv(path, header=None)
#             cols = ['uid', 'iid', 'y', 'pos_seq']
#             x_col = ['uid', 'iid']
#             y_col = ['y']
#             data.columns = cols
#             pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
#             pos_seq = torch.tensor(pos_seq, dtype=torch.long)
#             id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
#             X = torch.cat([id_fea, pos_seq], dim=1)
#             y = torch.tensor(data[y_col].values, dtype=torch.long)
#             if self.use_cuda:
#                 X = X.cuda()
#                 y = y.cuda()
#             dataset = TensorDataset(X, y)
#             data_iter = DataLoader(dataset, batchsize, shuffle=True)
#             return data_iter

#     def read_map_data(self):
#         cols = ['uid', 'iid', 'y', 'pos_seq']
#         data = pd.read_csv(self.meta_path, header=None)
#         data.columns = cols
#         X = torch.tensor(data['uid'].unique(), dtype=torch.long)
#         y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
#         if self.use_cuda:
#             X = X.cuda()
#             y = y.cuda()
#         dataset = TensorDataset(X, y)
#         data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
#         return data_iter

#     def read_aug_data(self):
#         cols_train = ['uid', 'iid', 'y']
#         x_col = ['uid', 'iid']
#         y_col = ['y']
#         src = pd.read_csv(self.src_path, header=None)
#         src.columns = cols_train
#         tgt = pd.read_csv(self.tgt_path, header=None)
#         tgt.columns = cols_train

#         X_src = torch.tensor(src[x_col].values, dtype=torch.long)
#         y_src = torch.tensor(src[y_col].values, dtype=torch.long)
#         X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
#         y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
#         X = torch.cat([X_src, X_tgt])
#         y = torch.cat([y_src, y_tgt])
#         if self.use_cuda:
#             X = X.cuda()
#             y = y.cuda()
#         dataset = TensorDataset(X, y)
#         data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

#         return data_iter
        
# data_src, data_tgt, data_meta, data_map, data_aug, data_test = self.get_data()
```

#### Code 펼치기

사용하는 Dataset을 하나하나 알아보자.

- **<code>data_src</code>**: Source Domain의 Dataset 이다. 해당 dataset은 [uid, iid, y]형태로 이루워져 있다.
- **<code>data_meta</code>**: Meta Network를 학습하기 위한 Train Dataset 이다. 해당 dataset은 [uid, iid, y, interaction_item_history]형태로 이루워져 있다. (overlap user - test user)
- **<code>data_test</code>**: Test의 Dataset 이다. 해당 dataset은 [uid, iid, y, interaction_item_history]형태로 이루워져 있다.


```python
'''
해당 Function은 Pandas로 저장하다 보니 생긴 문제를 해결하기 위한 Function 이다.
column의 값으로 List를 넣고 저장하여도, 다시 불러오면 String 형태로 되어 있다.
해당 값은 [iid1, iid2, ..., iidn]형색으로 되어있다.
따라서 아래와 같은 형태로 preprocessing 진행

공통사항
1. [, ] 을 제거한다.
2. , 으로 split하여 iid를 개별적으로 분리 한다.확인한다.
3. 만약 interaction이 발생한 iid가 없으면 (값이 []), 예외처리로 114495(64443 + 50052)로 처리한다.

User-Item Interaction Set
1. 최대 20개 까지의 Interaction history를 사용하게 설정한다.
2. padding방법은 post로 부족한 만큼 뒤에 0을 붙이도록 구성한다.
'''
def seq_extractor(x):
    x = x.rstrip(']').lstrip('[').split(', ')
    for i in range(len(x)):
        try:
            x[i] = int(x[i])
        except:
            x[i] = iid_all
    return np.array(x)

def read_log_data(path, batchsize, history=False):
    # 만약 history가 없는 source domain dataset이면 아래와 같이 interaction set을 고려하지 않고, dataset을 구성한다.
    if not history:
        cols = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data = pd.read_csv(path, header=None)
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long)
        y = torch.tensor(data[y_col].values, dtype=torch.long)
        if use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return data_iter
        
    else:
        data = pd.read_csv(path, header=None)
        cols = ['uid', 'iid', 'y', 'pos_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data.columns = cols
        pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(seq_extractor), maxlen=20, padding='post')
        pos_seq = torch.tensor(pos_seq, dtype=torch.long)
        id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
        X = torch.cat([id_fea, pos_seq], dim=1)
        y = torch.tensor(data[y_col].values, dtype=torch.long)
        if use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, batchsize, shuffle=True)
        return data_iter
```


```python
data_src = read_log_data(src_path, batchsize_src)
data_meta = read_log_data(meta_path, batchsize_meta, history=True)
data_test = read_log_data(test_path, batchsize_test, history=True)
```


```python
print('Pandas Meta Dataset')
data_meta_pd = pd.read_csv(meta_path, header = None)
data_meta_pd.head(3)
```

    Pandas Meta Dataset





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>121259</td>
      <td>89148</td>
      <td>4.0</td>
      <td>[30583, 34272, 44894, 47698, 36000, 21764, 714...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37300</td>
      <td>89148</td>
      <td>5.0</td>
      <td>[30583, 18018, 30092, 20548, 41588, 9912, 1357...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41304</td>
      <td>89148</td>
      <td>5.0</td>
      <td>[30583, 16982, 41376, 17905, 6261, 38838, 1886...</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Tensor Meta Dataset')

for x,y in data_meta:
    break

print('uid: {}'.format(x[1][0]))
print('iid: {}'.format(x[1][1]))
print('interaction history: {}'.format(x[1][2:]))
print('label: {}'.format(y[1]))
```

    Tensor Meta Dataset
    uid: 178102
    iid: 56082
    interaction history: tensor([18392, 38976, 38194, 46697, 31363, 26914, 20820, 17745, 49035,  9995,
            12462,  4653,  2449, 29707, 40499, 28775,  3220, 11076, 45918, 20044])
    label: tensor([4])


아래 데이터 셋을 직접 확인해 보면, [uid, iid, seq_iid1, seq_iid2, ...., seq_iid20]으로 이루워진것을 알 수 있다. 부족한 부분은 Padding으로 채워져 있다.


```python
print('Matching Padnas Dataset')

s_data = data_meta_pd[(data_meta_pd[0] == int(x[1][0])) & (data_meta_pd[1] == int(x[1][1]))].copy()
s_data
```

    Matching Padnas Dataset





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>265739</th>
      <td>178102</td>
      <td>56082</td>
      <td>4.0</td>
      <td>[4922, 31095, 25205, 9222, 12420, 40229, 40447...</td>
    </tr>
  </tbody>
</table>
</div>



- **<code>data_map</code>**: <font color=red>해당 데이터 셋은 Source User Embedding -> Target User Embedding을 학습하기 위한 데이터 셋 이다. 문제는 현재 Overlap되는 User만 사용하였다는 것 이다. 따라서, Target User ID를 임의로 만들기 위하여여 <code>torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)</code>으로서 정의하였다.</font>


```python
def read_map_data():
    cols = ['uid', 'iid', 'y', 'pos_seq']
    data = pd.read_csv(meta_path, header=None)
    data.columns = cols
    X = torch.tensor(data['uid'].unique(), dtype=torch.long)
    y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
    if use_cuda:
        X = X.cuda()
        y = y.cuda()
        
    dataset = TensorDataset(X, y)
    data_iter = DataLoader(dataset, batchsize_map, shuffle=True)
    return data_iter
```


```python
data_map = read_map_data()
```

### Training

**Train Function**

- Loss Function: MSE Loss로 통일하였다. -> Prediction의 경우에도 1~5사이의 값으로서 나오게 Model을 훈련하였다.ㅡ


```python
def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
    print('Training Epoch {}:'.format(epoch + 1))
    model.train()
    for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        if mapping:
            src_emb, tgt_emb = model(X, stage)
            loss = criterion(src_emb, tgt_emb)
        else:
            pred = model(X, stage)
            loss = criterion(pred, y.squeeze().float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
```

**CDR Training Process**  

CDR Model을 학습하기 위한 과정을 크게 3가지로 구성되어 있다.

1. Embedding Training: Source Domain User, Item Embedding을 Embedding하는 과정이다. (CDR Prediction)
2. Domain Adaption: Source Domain User Embedding -> Target Domain User Embedding을 학습하는 과정이다. 만약, Source, Target User Embedding이 predefine되어 있다면, 해당 과정은 필요없다. **후에 Training방법을 바꾸는 작업시 Model 및 Training 코드에서 수정해야 하는 부분이다.**
3. PRUPCDR: meta network및 prediction을 진행하는 과정이다.

**주의햐여야 할 점**: <font color=red>해당 논문에서는 각 Step에 대하여 Training을 진행할 때, <code>model.train()</code>으로서 전체 model에 대하여 학습을 진행하게 된다. 걱정되는 부분은, 1 step마다, 고정되어야 할 module들이 있다는 것 이다. 즉, 1 step으로서 Embedding 학습이 끝나게 되고, 2 step으로서 mapping function을 학습 진행 할 때는 backpropagation이 source model user, item embedding에 영향을 미치면 안된다는 것 이다. 어느것이 맞다고 할 수 없으나, 학습을 통해 알아봐야 할 부분이다.</font>

-> 해당 논문대로 구현시, Embedding과 Mapping을 Initialization을 위하여 사용되었다고 할 수 있다.


```python
def CDR(self, model, data_src, data_map, data_meta, data_test, criterion, optimizer_src, optimizer_map, optimizer_meta):
    print('=====CDR Pretraining=====')
    for i in range(self.epoch):
        self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')
    print('==========EMCDR==========')
    for i in range(self.epoch):
        self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
        mae, rmse = self.eval_mae(model, data_test, stage='test_map')
        self.update_results(mae, rmse, 'emcdr')
        print('MAE: {} RMSE: {}'.format(mae, rmse))
    print('==========PTUPCDR==========')
    for i in range(self.epoch):
        self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_meta')
        mae, rmse = self.eval_mae(model, data_test, stage='test_meta')
        self.update_results(mae, rmse, 'ptupcdr')
        print('MAE: {} RMSE: {}'.format(mae, rmse))
```

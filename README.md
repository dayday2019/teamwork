## “心跳信号预测”项目报告

### 1.问题提出


赛题以心电图数据为背景，要求选手根据心电图数据预测心跳信号，其中心跳信号对应正常病例以及受不同心律不齐和心肌梗塞影响的病例，这是一个多分类问题。

数据来自某平台心电图数据记录，总数据量超过20万，主要为1列心跳信号序列数据，其中每个样本的信号序列采样频次一致，长度相等。数据集字段表如下图所示：

![AzflB.png](https://s1.328888.xyz/2022/04/30/AzflB.png)

并从中抽取了10万条作为训练集，2万条作为测试集。

### 2.数据分析和可视化
通过以上内容，我们了解了赛题信息及数据格式，接下来我们对数据进一步分析。主要是探索性分析和描述性分析，以此了解变量间的相互关系以及变量与预测值之间存在的关系。

首先查看train和test数据集首尾数据，对数据有一个整体认识
![AzAT7.png](https://s1.328888.xyz/2022/04/30/AzAT7.png)

![Azq0C.png](https://s1.328888.xyz/2022/04/30/Azq0C.png)
为方便掌握数据的大概范围，用describe函数查看各列的统计量，个数count、平均值mean、方差std、最小值min、中位数25%、50%、75%以及最大值。

![Azttt.png](https://s1.328888.xyz/2022/04/30/Azttt.png)

info函数查看各列的数据类型，判断是否有异常数据，以及是否存在Nan值，分析后得知，训练集和测试集均不存在缺失值和异常值，非常理想。
![AzpTP.png](https://s1.328888.xyz/2022/04/30/AzpTP.png)

![AzTgi.png](https://s1.328888.xyz/2022/04/30/AzTgi.png)

查看数据的整体分布情况
![AzbZQ.png](https://s1.328888.xyz/2022/04/30/AzbZQ.png)

查看偏度（skewness）以及峰度（kurtosis）。分析可知，偏度 (skewness) 大于 0，表示数据分布倾向于右偏，长尾在右；峰度 (kurtosis) 小于 0，表示数据分布与正态分布相比，较为平坦，为平顶峰。
![AzVg3.png](https://s1.328888.xyz/2022/04/30/AzVg3.png)

### 3.模型探索

#### 3.1LightGBM
##### 3.1.1选取原因
基于数据集特点选取具有利用弱分类器（决策树）迭代训练以得到最优模型的LGBM模型进行数据挖掘，该模型具有训练效果好、不易过拟合，在多分类问题上表现很好等优点。该模型也可以很好地利用在特征提取阶段得到的时间序列特征。
##### 3.1.2评价标准
F1分数（F1 Score），是统计学中用来衡量二分类模型精确度的一种指标。它同时兼顾了分类模型的精确率和召回率。F1分数可以看作是模型精确率和召回率的一种调和平均，它的最大值是1，最小值是0。
##### 3.1.3实验结果
LightGBM在验证集上的f1:0.965
#### 3.2ResNet50

##### 3.2.1选取原因
选择使用ResNet50模型进行数据挖掘，ResNet50模型是CNN模型的一种，常用于对图像数据进行处理，选择它的原因有两点：    
1、由于心跳信号是一组有序的数据，其中单独的一个信号往往与其相邻的信号存在一定的联系，因此使用ResNet50这种CNN模型对心跳信号数据进行处理是较为合理的，可以充分发挥CNN所具有的局部感知能力，即它的每一次卷积运算都会考虑多个相邻的信号。    
2、ResNet50模型具有残差结构，容易优化，并且能够通过增加网络深度来提高准确率。

##### 3.2.2模型结构

我们所使用的Resnet50 网络中包含了 49 个卷积层和2个全连接层。Resnet50网络结构可以分成七个部分，第一部分不包含残差块，主要对输入进行卷积、正则化、激活函数、最大池化的计算。第二、三、四、五部分结构都包含了残差块，每个残差块都有三层卷积，网络总共有1+3×(3+4+6+3)=49个卷积层，加上最后的全连接层总共是 51 层。网络的输入为batch_size×1×205，输出为batch_size×4。此外，为了使原本用于图像等二维数据的ResNet50模型能够适应心跳信号这种一维数据，我们将模型中的所有二维卷积层，二维池化层以及二维BatchNorm层改为了一维的形式。

##### 3.2.3评价标准

1、平均指标：针对某一信号，若真实值为[y_1,y_2,y_3,y_4]，模型预测概率值为[a_1,a_2,a_3,a_4]那么该模型的平均指标abs为

![](https://latex.codecogs.com/svg.image?abs=\sum_{j=1}^{n}\sum_{i=1}^{4}\left|y_i-a_i\right|)

例如，心跳信号为1，会通过编码转成[0, 1, 0, 0]，预测不同心跳信号概率为[0.1, 0.7, 0.1, 0.1]，那么这个预测结果的平均指标为。

![](https://latex.codecogs.com/svg.image?abs=\left|0.1-0\right|&plus;\left|0.7-1\right|&plus;\left|0.1-0\right|&plus;\left|0.1-0\right|=0.6)

2、预测准确率：若预测结果与真实值一致，认为预测正确，否则认为预测不正确，那么该模型的预测准确率为

![](https://latex.codecogs.com/svg.image?acc=\frac{Number\&space;of\&space;correct\&space;prediction\&space;results}{Total\&space;number\&space;of\&space;samples})

##### 3.2.4实验结果
准确率：0.9872  
平均指标：679.14
#### 3.3CNN
##### 3.3.1选取原因
心跳信号是一个典型的时间序列问题，而处理时序问题的常用方法就是一维卷积。所以，我们进行了相应的尝试。
##### 3.3.2模型结构
一维卷积和最常见的二维卷积并不是指输入数据的形状是一维或者二维，而是指卷积核的移动方向。例如，二维的卷积核可以在两个方向上移动，而一维的卷积核只能在一个方向上移动来提取数据特征。所以，二维卷积运算后输出的是一个矩阵，而一维卷积输出的是一个向量。
模型使用九层的神经网络结构，包括六层卷积层，三层全连接层：

![](https://github.com/dayday2019/teamwork/blob/main/images/1.png?raw=true)

在偶数层，也就是第2层，第4层，第6层之后加入池化层。池化层最开始使用平均池化，训练集中的最小score稳定在230左右。模型表现良好，但是不足以进入长期赛榜单。所以决定尝试使用最大池化来进一步提升神经网络提取特征的能力。
![](https://github.com/dayday2019/teamwork/blob/main/images/2.png?raw=true)

后续也可以采用平均池化和最大池化交替使用的方式。对于全连接层来说，由于过深的神经网络很容易出现过拟合现象，所以加入正则化方法，在每层全连接层之后加入dropout层，dropout率设为0.5。
##### 3.3.3训练参数
模型训练时将训练集二八分为测试集和训练集，优化器选择：Adam，损失函数选择： 交叉熵损失，Batch size 为64，初始学习率 1e-5，训练100轮。
##### 3.3.4模型结果
将池化层修改为最大池化之后的预测模型在长期赛中score降低到209，进入了前200名。
![](https://github.com/dayday2019/teamwork/blob/main/images/3.png?raw=true)

##### 3.3.5结果提升：
由于本赛题属于分类问题，可使用阈值法将预测概率不小于 0.5 的类别置 1，其余则置 0。对于预测概率均小于 0.5 的难样本，使用二次处理：若最大预测值比次大预测值至少高 0.05，则认为最大预测值足够可信并置 1 其余置 0；否则认为最大预测值和次大预测值区分度不够高，难以分辨不作处理，仅令最小的另外两个预测值置 0。在对初始预测值进行阈值法处理后，预测效果有所提升，最终排名168名。
![](https://github.com/dayday2019/teamwork/blob/main/images/4.png?raw=true)

### 4.LightGBM实验过程

### 引入需要的包


```python
import os
import gc
import math

import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler


from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')
```

### 引入数据预处理后的训练集train.csv和测试集testA.csv数据如下，将两个数据集的表头打印如下


```python
train = pd.read_csv('E:\\master\\dataMing\\train.csv')
test=pd.read_csv('E:\\master\\dataMing\\testA.csv')
train.head()
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
      <th>id</th>
      <th>heartbeat_signals</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.9912297987616655,0.9435330436439665,0.764677...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.9714822034884503,0.9289687459588268,0.572932...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1.0,0.9591487564065292,0.7013782792997189,0.23...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.9757952826275774,0.9340884687738161,0.659636...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0,0.055816398940721094,0.26129357194994196,0...</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
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
      <th>id</th>
      <th>heartbeat_signals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100000</td>
      <td>0.9915713654170097,1.0,0.6318163407681274,0.13...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001</td>
      <td>0.6075533139615096,0.5417083883163654,0.340694...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002</td>
      <td>0.9752726292239277,0.6710965234906665,0.686758...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100003</td>
      <td>0.9956348033996116,0.9170249621481004,0.521096...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100004</td>
      <td>1.0,0.8879490481178918,0.745564725322326,0.531...</td>
    </tr>
  </tbody>
</table>
</div>



### 定义一个减少内存占用的函数，这其中，第三行打印中，有**{:.2f}**，这个是指这部分填写后面format(start_mem)的内容，2是指保存两位小数，f是指float类型。


```python
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
```

### 对进行过特征处理的数据进行简单的预处理，主要是将数据格式化和去除空值，将处理后的数据打印展示如下


```python
# 简单预处理
train_list = []

for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_'+str(i) for i in range(len(train_list[0])-2)] + ['label']
train = reduce_mem_usage(train)

test_list=[]
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]
test = reduce_mem_usage(test)
```

    Memory usage of dataframe is 157.93 MB
    Memory usage after optimization is: 39.67 MB
    Decreased by 74.9%
    Memory usage of dataframe is 31.43 MB
    Memory usage after optimization is: 7.90 MB
    Decreased by 74.9%



```python
train.head()
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
      <th>id</th>
      <th>s_0</th>
      <th>s_1</th>
      <th>s_2</th>
      <th>s_3</th>
      <th>s_4</th>
      <th>s_5</th>
      <th>s_6</th>
      <th>s_7</th>
      <th>s_8</th>
      <th>...</th>
      <th>s_196</th>
      <th>s_197</th>
      <th>s_198</th>
      <th>s_199</th>
      <th>s_200</th>
      <th>s_201</th>
      <th>s_202</th>
      <th>s_203</th>
      <th>s_204</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.991211</td>
      <td>0.943359</td>
      <td>0.764648</td>
      <td>0.618652</td>
      <td>0.379639</td>
      <td>0.190796</td>
      <td>0.040222</td>
      <td>0.026001</td>
      <td>0.031708</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.971680</td>
      <td>0.929199</td>
      <td>0.572754</td>
      <td>0.178467</td>
      <td>0.122986</td>
      <td>0.132324</td>
      <td>0.094421</td>
      <td>0.089600</td>
      <td>0.030487</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.000000</td>
      <td>0.958984</td>
      <td>0.701172</td>
      <td>0.231812</td>
      <td>0.000000</td>
      <td>0.080688</td>
      <td>0.128418</td>
      <td>0.187500</td>
      <td>0.280762</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.975586</td>
      <td>0.934082</td>
      <td>0.659668</td>
      <td>0.249878</td>
      <td>0.237061</td>
      <td>0.281494</td>
      <td>0.249878</td>
      <td>0.249878</td>
      <td>0.241455</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.000000</td>
      <td>0.055817</td>
      <td>0.261230</td>
      <td>0.359863</td>
      <td>0.433105</td>
      <td>0.453613</td>
      <td>0.499023</td>
      <td>0.542969</td>
      <td>0.616699</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 207 columns</p>
</div>




```python
test.head()
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
      <th>id</th>
      <th>s_0</th>
      <th>s_1</th>
      <th>s_2</th>
      <th>s_3</th>
      <th>s_4</th>
      <th>s_5</th>
      <th>s_6</th>
      <th>s_7</th>
      <th>s_8</th>
      <th>...</th>
      <th>s_195</th>
      <th>s_196</th>
      <th>s_197</th>
      <th>s_198</th>
      <th>s_199</th>
      <th>s_200</th>
      <th>s_201</th>
      <th>s_202</th>
      <th>s_203</th>
      <th>s_204</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100000.0</td>
      <td>0.991699</td>
      <td>1.000000</td>
      <td>0.631836</td>
      <td>0.136230</td>
      <td>0.041412</td>
      <td>0.102722</td>
      <td>0.120850</td>
      <td>0.123413</td>
      <td>0.107910</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100001.0</td>
      <td>0.607422</td>
      <td>0.541504</td>
      <td>0.340576</td>
      <td>0.000000</td>
      <td>0.090698</td>
      <td>0.164917</td>
      <td>0.195068</td>
      <td>0.168823</td>
      <td>0.198853</td>
      <td>...</td>
      <td>0.389893</td>
      <td>0.386963</td>
      <td>0.367188</td>
      <td>0.364014</td>
      <td>0.360596</td>
      <td>0.357178</td>
      <td>0.350586</td>
      <td>0.350586</td>
      <td>0.350586</td>
      <td>0.36377</td>
    </tr>
    <tr>
      <th>2</th>
      <td>100002.0</td>
      <td>0.975098</td>
      <td>0.670898</td>
      <td>0.686523</td>
      <td>0.708496</td>
      <td>0.718750</td>
      <td>0.716797</td>
      <td>0.720703</td>
      <td>0.701660</td>
      <td>0.596680</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100003.0</td>
      <td>0.995605</td>
      <td>0.916992</td>
      <td>0.520996</td>
      <td>0.000000</td>
      <td>0.221802</td>
      <td>0.404053</td>
      <td>0.490479</td>
      <td>0.527344</td>
      <td>0.518066</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100004.0</td>
      <td>1.000000</td>
      <td>0.888184</td>
      <td>0.745605</td>
      <td>0.531738</td>
      <td>0.380371</td>
      <td>0.224609</td>
      <td>0.091125</td>
      <td>0.057648</td>
      <td>0.003914</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 206 columns</p>
</div>



### 准备训练数据和测试数据，x_train为训练数据，x_test为测试数据


```python
x_train = train.drop(['id','label'], axis=1)
y_train = train['label']
x_test=test.drop(['id'], axis=1)
```

### 定义LGBM的损失函数


```python
def abs_sum(y_pre,y_tru):
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(sum(abs(y_pre-y_tru)))
    return loss
```

### 模型训练，lgb_model模型通过cv_model训练增长得到


```python
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5
    seed = 2021
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test = np.zeros((test_x.shape[0],4))

    cv_scores = []
    onehot_encoder = OneHotEncoder(sparse=False)
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
        if clf_name == "lgb":
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'multiclass',
                'num_class': 4,
                'num_leaves': 2 ** 5,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 4,
                'learning_rate': 0.1,
                'seed': seed,
                'nthread': 28,
                'n_jobs':24,
                'verbose': -1,
            }

            model = clf.train(params, 
                      train_set=train_matrix, 
                      valid_sets=valid_matrix, 
                      num_boost_round=2000, 
                      verbose_eval=100, 
                      early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration) 
            
        val_y=np.array(val_y).reshape(-1, 1)
        val_y = onehot_encoder.fit_transform(val_y)
        print('预测的概率矩阵为：')
        print(test_pred)
        test += test_pred
        score=abs_sum(val_y, val_pred)
        cv_scores.append(score)
        print(cv_scores)
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    test=test/kf.n_splits

    return test
```


```python
def lgb_model(x_train, y_train, x_test):
    lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_test
```


```python
lgb_test = lgb_model(x_train, y_train, x_test)
```

    ************************************ 1 ************************************
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 200 rounds
    [100]	valid_0's multi_logloss: 0.0525735
    [200]	valid_0's multi_logloss: 0.0422444
    [300]	valid_0's multi_logloss: 0.0407076
    [400]	valid_0's multi_logloss: 0.0420398
    Early stopping, best iteration is:
    [289]	valid_0's multi_logloss: 0.0405457
    预测的概率矩阵为：
    [[9.99969791e-01 2.85197261e-05 1.00341946e-06 6.85357631e-07]
     [7.93287264e-05 7.69060914e-04 9.99151590e-01 2.00810971e-08]
     [5.75356884e-07 5.04051497e-08 3.15322414e-07 9.99999059e-01]
     ...
     [6.79267940e-02 4.30206297e-04 9.31640185e-01 2.81516302e-06]
     [9.99960477e-01 3.94098074e-05 8.34030725e-08 2.94638661e-08]
     [9.88705846e-01 2.14081630e-03 6.67418381e-03 2.47915423e-03]]
    [607.0736049372185]
    ************************************ 2 ************************************
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 200 rounds
    [100]	valid_0's multi_logloss: 0.0566626
    [200]	valid_0's multi_logloss: 0.0450852
    [300]	valid_0's multi_logloss: 0.044078
    [400]	valid_0's multi_logloss: 0.0455546
    Early stopping, best iteration is:
    [275]	valid_0's multi_logloss: 0.0437793
    预测的概率矩阵为：
    [[9.99991401e-01 7.69109547e-06 6.65504756e-07 2.42084688e-07]
     [5.72380482e-05 1.32812809e-03 9.98614607e-01 2.66534396e-08]
     [2.82123411e-06 4.13195205e-07 1.34026965e-06 9.99995425e-01]
     ...
     [6.96398024e-02 6.52459907e-04 9.29685742e-01 2.19960932e-05]
     [9.99972366e-01 2.75069005e-05 7.68142933e-08 5.07415018e-08]
     [9.67263676e-01 7.26154408e-03 2.41533542e-02 1.32142531e-03]]
    [607.0736049372185, 623.4313863731124]
    ************************************ 3 ************************************
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 200 rounds
    [100]	valid_0's multi_logloss: 0.0498722
    [200]	valid_0's multi_logloss: 0.038028
    [300]	valid_0's multi_logloss: 0.0358066
    [400]	valid_0's multi_logloss: 0.0361478
    [500]	valid_0's multi_logloss: 0.0379597
    Early stopping, best iteration is:
    [340]	valid_0's multi_logloss: 0.0354344
    预测的概率矩阵为：
    [[9.99972032e-01 2.62406774e-05 1.17282152e-06 5.54230651e-07]
     [1.05242811e-05 6.50215805e-05 9.99924453e-01 6.93812546e-10]
     [1.93240868e-06 1.10384984e-07 3.76773426e-07 9.99997580e-01]
     ...
     [1.34894410e-02 3.84569683e-05 9.86471555e-01 5.46564350e-07]
     [9.99987431e-01 1.25532882e-05 1.03902298e-08 5.46727770e-09]
     [9.78722948e-01 1.06329839e-02 6.94192038e-03 3.70214810e-03]]
    [607.0736049372185, 623.4313863731124, 508.02381607269535]
    ************************************ 4 ************************************
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 200 rounds
    [100]	valid_0's multi_logloss: 0.0564768
    [200]	valid_0's multi_logloss: 0.0448698
    [300]	valid_0's multi_logloss: 0.0446719
    [400]	valid_0's multi_logloss: 0.0470399
    Early stopping, best iteration is:
    [250]	valid_0's multi_logloss: 0.0438853
    预测的概率矩阵为：
    [[9.99979692e-01 1.70821979e-05 1.27048476e-06 1.95571841e-06]
     [5.66207785e-05 4.02275314e-04 9.99541086e-01 1.82828519e-08]
     [2.62267451e-06 3.58613522e-07 4.78645006e-06 9.99992232e-01]
     ...
     [4.56636552e-02 5.69497433e-04 9.53758468e-01 8.37980573e-06]
     [9.99896785e-01 1.02796802e-04 2.46636563e-07 1.72061021e-07]
     [8.70911669e-01 1.73790185e-02 1.04478175e-01 7.23113697e-03]]
    [607.0736049372185, 623.4313863731124, 508.02381607269535, 660.4867407547266]
    ************************************ 5 ************************************
    [LightGBM] [Warning] num_threads is set with nthread=28, will be overridden by n_jobs=24. Current value: num_threads=24
    Training until validation scores don't improve for 200 rounds
    [100]	valid_0's multi_logloss: 0.0506398
    [200]	valid_0's multi_logloss: 0.0396422
    [300]	valid_0's multi_logloss: 0.0381065
    [400]	valid_0's multi_logloss: 0.0390162
    [500]	valid_0's multi_logloss: 0.0414986
    Early stopping, best iteration is:
    [324]	valid_0's multi_logloss: 0.0379497
    预测的概率矩阵为：
    [[9.99993352e-01 6.02902202e-06 1.13002685e-07 5.06277302e-07]
     [1.03959552e-05 5.03778956e-04 9.99485820e-01 5.07638601e-09]
     [1.92568065e-07 5.07155306e-08 4.94690856e-08 9.99999707e-01]
     ...
     [8.83103121e-03 2.51969353e-05 9.91142776e-01 9.96143937e-07]
     [9.99984791e-01 1.51997858e-05 5.62426491e-09 3.80450197e-09]
     [9.86084001e-01 8.75968498e-04 1.09742304e-02 2.06580027e-03]]
    [607.0736049372185, 623.4313863731124, 508.02381607269535, 660.4867407547266, 539.2160054696064]
    lgb_scotrainre_list: [607.0736049372185, 623.4313863731124, 508.02381607269535, 660.4867407547266, 539.2160054696064]
    lgb_score_mean: 587.6463107214719
    lgb_score_std: 55.944536405714565



```python
lgb_test
```




    array([[9.99981254e-01, 1.71125438e-05, 8.45046636e-07, 7.88733736e-07],
           [4.28215579e-05, 6.13652971e-04, 9.99343511e-01, 1.41575174e-08],
           [1.62884845e-06, 1.96662878e-07, 1.37365693e-06, 9.99996801e-01],
           ...,
           [4.11101448e-02, 3.43163508e-04, 9.58539745e-01, 6.94675406e-06],
           [9.99960370e-01, 3.94933168e-05, 8.45736848e-08, 5.23076338e-08],
           [9.58337628e-01, 7.65806626e-03, 3.06443728e-02, 3.35993298e-03]])



### 得到数据集的概率模型


```python
temp=pd.DataFrame(lgb_test)
temp
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.999981</td>
      <td>1.711254e-05</td>
      <td>8.450466e-07</td>
      <td>7.887337e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000043</td>
      <td>6.136530e-04</td>
      <td>9.993435e-01</td>
      <td>1.415752e-08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000002</td>
      <td>1.966629e-07</td>
      <td>1.373657e-06</td>
      <td>9.999968e-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999970</td>
      <td>1.909713e-05</td>
      <td>1.097002e-05</td>
      <td>3.576703e-08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.999983</td>
      <td>1.769712e-06</td>
      <td>1.482817e-05</td>
      <td>1.966254e-07</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>19995</th>
      <td>0.998096</td>
      <td>3.060176e-04</td>
      <td>1.085313e-04</td>
      <td>1.489757e-03</td>
    </tr>
    <tr>
      <th>19996</th>
      <td>0.999846</td>
      <td>1.436305e-04</td>
      <td>1.074898e-05</td>
      <td>8.837766e-08</td>
    </tr>
    <tr>
      <th>19997</th>
      <td>0.041110</td>
      <td>3.431635e-04</td>
      <td>9.585397e-01</td>
      <td>6.946754e-06</td>
    </tr>
    <tr>
      <th>19998</th>
      <td>0.999960</td>
      <td>3.949332e-05</td>
      <td>8.457368e-08</td>
      <td>5.230763e-08</td>
    </tr>
    <tr>
      <th>19999</th>
      <td>0.958338</td>
      <td>7.658066e-03</td>
      <td>3.064437e-02</td>
      <td>3.359933e-03</td>
    </tr>
  </tbody>
</table>
<p>20000 rows × 4 columns</p>
</div>



### K折交叉验证，folds可以设置是几折，seed是随机种子


```python
from sklearn.model_selection import KFold
x_train = train.drop(['id','label'], axis=1)
y_train = train['label']
folds = 5
seed = 2022
kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
```

### f1得分，传入真实值和预测值即可算出F1得分，进而判断效果


```python
from sklearn.metrics import f1_score
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(4, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'f1_score', score_vali, True

```

### 调参，并在测试集上验证得到的模型的效果


```python
from sklearn.model_selection import train_test_split
import lightgbm as lgb
x_train_split, x_val, y_train_split, y_val = train_test_split(x_train, y_train, test_size=0.2)
train_matrix = lgb.Dataset(x_train_split, label=y_train_split)
valid_matrix = lgb.Dataset(x_val, label=y_val)

params = {
    "learning_rate": 0.1,
    "boosting": 'gbdt',  
    "lambda_l2": 0.1,
    "max_depth": -1,
    "num_leaves": 128,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "metric": None,
    "objective": "multiclass",
    "num_class": 4,
    "nthread": 10,
    "verbose": -1,
}

"""使用训练集数据进行模型训练"""
model = lgb.train(params, 
                  train_set=train_matrix, 
                  valid_sets=valid_matrix, 
                  num_boost_round=2000, 
                  verbose_eval=50, 
                  early_stopping_rounds=200,
                  feval=f1_score_vali )
```

    Training until validation scores don't improve for 200 rounds
    [50]	valid_0's multi_logloss: 0.0472006	valid_0's f1_score: 0.956668
    [100]	valid_0's multi_logloss: 0.0402238	valid_0's f1_score: 0.965285
    [150]	valid_0's multi_logloss: 0.041574	valid_0's f1_score: 0.967649
    [200]	valid_0's multi_logloss: 0.0432073	valid_0's f1_score: 0.968158
    [250]	valid_0's multi_logloss: 0.0444573	valid_0's f1_score: 0.968771
    Early stopping, best iteration is:
    [99]	valid_0's multi_logloss: 0.0402124	valid_0's f1_score: 0.965188



```python
val_pre_lgb = model.predict(x_val, num_iteration=model.best_iteration)
preds = np.argmax(val_pre_lgb, axis=1)
score = f1_score(y_true=y_val, y_pred=preds, average='macro')
print('lightgbm单模型在验证集上的f1：{}'.format(score))
```

    lightgbm单模型在验证集上的f1：0.9651880671197886

### 5.挖掘结果及总结
本小组选择心跳信号的数据集，首先对其进行数据分析，初步分析数据的特点，然后分别使用LightGBM、ResNet50、CNN模型对其进行数据挖掘，得到了不同的预测结果：   
1、使用LightGBM模型在验证集上f1分数达到了0.965  
2、使用ResNet50模型预测结果的平均指标为679.14，准确率为0.9872  
3、使用CNN模型在长期赛中score降低到了206，最终排名168名  
后续，我们计划进一步挖掘数据特征，做好特征工程，以便使我们的模型得到更好的效果。此外，针对LightGBM和CNN模型，我们计划采用均值法、投票法等模型融合的方法进一步提升预测效果。



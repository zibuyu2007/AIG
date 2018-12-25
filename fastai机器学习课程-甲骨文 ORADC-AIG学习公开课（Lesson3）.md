>*Hi，这是我们第三次公开课。之所以有这个分享课程是因为大家太忙（懒），没有时间看fastai在线视频和笔记。而且视频和笔记都是英文的，大家也不想费脑子（懒）。所以本课程的目的就是把Jeremy老师的视频用中文再给大家讲一遍，另外把Hiromi小姐的笔记翻译加工一下分享给大家。*

# 随机森林模型解析
### [Kaggle](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

>**通过机器学习更好地理解数据**
这种想法与通常的见解相左，类如随机森林这样的算法是黑盒的、不能窥其实质的。事实恰恰相反，与传统方法相比，随机森林允许我们更深入、更快速地理解数据。

我们这次要探究更大的数据集，利用kaggle的货物销售预测竞赛（Grocery Forecasting），其中包含了超过100万行的数据集。
>**什么时候使用随机森林?**
>Jeremy说想不出有什么东西是一点用没有的，总是值得试一试随机森林。所以真正的问题可能是在什么情况下我们还应该尝试其他东西。简单地说，对于非结构化数据(图像、声音等)，几乎肯定要使用深度学习；而对于协同过滤模型(杂货竞争就是这种类型)，无论是随机森林还是深度学习方法都不是你想要的，你需要做一些调整。

## 1. 法沃里塔公司货物销售预测(Corporación Favorita Grocery Sales Forecasting)

让我们在处理一个非常大的数据集时，基本上和之前的处理过程相同。但是有一些情况下我们不能使用默认值，因为默认值运行得太慢了。
>能够解释你正在处理的问题是很重要的。在机器学习问题中需要理解的关键是:
自变量是什么?
因变量是什么（我们要预测的东西）?

在这个货物销售竞赛中：
因变量——在两周的时间内，每个商店每天销售多少种产品。
自变量——在过去的几年里，每家商店每天卖出多少件产品。对于每个商店，它的位置以及它是什么类型的商店(元数据)。对于每种类型的产品，它是什么类型的产品，等等。对于每个日期，我们都有元数据，例如油价是什么。
这就是我们所说的**关系数据集**。关系数据集是一种我们可以将许多不同的信息片段连接在一起的数据集。具体来说，这种关系数据集就是我们所说的“星型模式”，其中有一些中央事务表。
在这个竞赛中，中央事务表是train.csv，其包含按日期、store_nbr和item_nbr出售的单位数。我们可以从这里连接各种元数据(因此得名“星型”模式——也称为“雪花”模式)。

### 1.1 读取数据
```python
types = {'id': 'int64',
         'item_nbr': 'int32',
         'store_nbr': 'int8',
         'unit_sales': 'float32',
         'onpromotion': 'object'}
%time df_all = pd.read_csv(f'../input/train.csv', parse_dates=['date'], dtype=types, infer_datetime_format=True)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222163047791.png)
 - 如果设置``low_memory=False``，无论有多少内存，它都会耗尽内存。
 - 为了限制在读取时占用的空间，我们为每个列名创建一个字典，对应值为该列的数据类型。通过在数据集上运行```less```或```head```，你将决定采用哪种数据类型。
- 通过这些调整，我们可以在2分钟的时间内读取125,497,040行。
- Python本身并不快，但是我们在数据科学中想要做的几乎所有事情都是用C或者Cython语言写的。在panda中，许多代码是用经过大量优化的汇编语言编写的。在幕后，很多工作都是调用基于Fortran的线性代数库。

>指定int64和int是否有性能考虑?
>这里的关键是使用尽可能少的位来完全表示列的值。如果我们对item_nbr使用int8，因为item_nbr的最大值大于255，所以这是不合适的。另一方面，如果我们对store_nbr使用int64，那么它使用的bit就比需要的要多。这样考虑的目的就是避免耗尽RAM。当使用大型数据集时，通常会发现性能瓶颈是读写RAM，而不是CPU操作。根据经验来说，较小的数据类型通常运行得更快，特别是如果可以使用单个指令多数据(Single Instruction Multiple Data，SIMD)向量化代码，它可以将更多的数字打包到单个向量中，以便一次运行。

>不需要对数据重新洗牌么?
>通过使用UNIX命令```shuf```，可以在命令提示符处获得一个随机的数据样本，然后就可以读取它了。这是一个很好的方法，例如，找出要使用的数据类型——从一个随机样本中读取数据，然后让panda为您计算出来。一般来说，Jeremy会在一个样本上做尽可能多的工作，直到他确信已经理解了这个样本才会继续。

要使用``shuf``从文件中随机选择一行，请使用-n选项。这将输出限制为指定的数字。你也可以指定输出文件:
```cmd
shuf -n 5 -o sample_training.csv train.csv
```
'onpromotion': ‘object'——object是一种通用的Python数据类型，速度慢，占内存大。这里使用它因为onpromotion是一个布尔值，而且有缺失值，所以我们需要先处理这个问题，然后才能把它变成布尔值，如下所示：
```python
df_all.onpromotion.fillna(False, inplace=True)
df_all.onpromotion = df_all.onpromotion.map({'False': False, 
                                             'True': True})
df_all.onpromotion = df_all.onpromotion.astype(bool)
```
- ``fillna(False)``:在经过检查数据后，我们决定用False填充缺失值。
- ``map({'False': False， 'True': True})``:object通常以字符串的形式读入，所以用实际的布尔值替换字符串'True'和'False'。
- ``astype(bool)``:最后将其转换为boolean类型。

pandas一般都很快，所以你可以在几十秒内总结出所有1.25亿条记录中的每一列:
```python
%time df_all.describe(include='all')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222171730613.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222171910152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

- 先看日期。日期之所以重要，是因为你在实践中使用的任何模型，将会应用到某个日期，而且这个日期要比训练模型用到的日期要晚。如果世界上有任何变化，你也需要知道预测精度是如何变化的。因此，对于Kaggle或你自己的项目，应该始终确保日期不重叠。
- 在这种情况下，使用从2013年到2017年8月训练集数据。
接下来处理测试集。
```python
df_test = pd.read_csv(f'{PATH}test.csv', parse_dates = ['date'],
                      dtype=types, infer_datetime_format=True)
df_test.onpromotion.fillna(False, inplace=True)
df_test.onpromotion = df_test.onpromotion.map({'False': False, 
                                               'True': True})
df_test.onpromotion = df_test.onpromotion.astype(bool)
df_test.describe(include='all')
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222173155478.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
- 这是一个关键问题——在你理解这个基本部分之前，无法做任何有用的机器学习。即你有四年的数据，然后试图预测未来两周的结果。
- 如果你想使用较小的数据集，我们应该使用最近的数据集，而不是随机集。

>四年前同样的时间段(比如圣诞节前后)不是很重要吗?
>完全正确。这并不是说四年前的信息没用，所以我们不想完全抛弃这些数据。但作为第一步，如果你想提交个均值，你应该不会提交2012年的销售均值，而是可能会提交上个月的销售均值。稍后，我们可能想要更重视最近的日期，因为它们可能更相关。但我们应该做一些探索性的数据分析来验证这一点。
显示最底下的数据，也就是离现在最近的数据。
```python
df_all.tail()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222173632972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

### 1.2 处理数据
```python
df_all.unit_sales = np.log1p(np.clip(df_all.unit_sales, 0, None))
```
- 我们需要对销售额取对数，因为我们试图预测一些根据比率变化的东西，此外，均方根对数误差也是这个竞赛中用于评价的标准。
- ``np.clip (df_all.unit_sales, 0, None):``有一些销售额是负数（代表退货），竞赛中说明了应该将其视为零。``clilp``截断到指定的最小值和最大值。
- ``np.log1p``:值加1的对数。竞赛说明其会使用均方根对数 + 1误差，因为log(0)没有意义。

在这里我们只用不到1/5的数据，否则kernel会崩啊。
```python
lenall = len(df_all)
df_all = df_all[(lenall//5)*4:]
%time add_datepart(df_all, 'date')
```
我们可以像往常一样添加日期部分。这需要几分钟的时间，所以我们应该先在sample上运行所有这些，以确保它能正常工作。直到你确定各种设定都是合理的了，就可以去运行整个集合了。

接下来划分验证集和训练集。我们把验证集设定的和测试机一样大小。
```python
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid = len(df_test)
n_trn = len(df_all) - n_valid
train, valid = split_vals(df_all, n_trn)
train.shape, valid.shape
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222191621747.png)
>我们不需要运行``train_cats``或``apply_cats``，因为所有的数据类型都已经是数字了(注意，``apply_cats``对验证集应用的分类代码与培训集相同)

调用``proc_df``检查缺失的值等等。``%%time``可以监测多行程序的运行时间。
```python
%%time
trn, y, nas  = proc_df(train, 'unit_sales')
val, y_val, nas = proc_df(valid, 'unit_sales', nas)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222191648483.png)
### 1.3 训练模型
```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())    
set_rf_samples(1_000_000)

%time x = np.array(trn, dtype=np.float32)
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=100, 
                          n_jobs=8)
%time m.fit(x, y)
```
我们可能不想从1.25亿条记录中创建一个树(不知道需要多长时间)。你可以从10k或100k开始，算出能跑多少量。数据集的大小与构建随机森林的时间没有关系，影响构建时间的是estimators乘以样本大小。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222203447523.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
>``n_jobs``是什么?
>之前我们用``n_jobs=-1``。``n_jobs``是要使用的内核的数量。Jeremy在一台大约有60个核的电脑上运行这个程序，所以如果使用所有的核速度反而会变慢。

>``x = np.array(trn, dtype=np.float32)``:这将把data frame转换成浮点数的数组。在随机森林代码中，原本也会这么做。但是假设我们想要运行一些具有不同超参数的随机森林，我们花点时间自己先做一次，以后每次运行``fit``时就可以节省相同的时间。

如果运行代码花费了相当长的时间，可以将%prun放在前面，用以打印内部花费时间的明细情况。
```python
%prun m.fit(x, y)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222203514879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

- 这将运行一个分析器，并告诉你哪行代码花费的时间最多。这里是scikit-learn中将data frame转换为numpy数组的代码行。
- 查看哪些事情占用了时间称为“分析”，在软件工程中，这是最重要的工具之一。（但数据科学家们往往低估了这一点。）
- 为了好玩，请尝试在需要10-20秒的代码上运行``%prun``，看看是否可以学着解释和使用分析器输出。
- 在分析器中执行``set_rf_samples``时，我们不能使用OOB得分，因为如果这样做，它将使用其他1.24亿行来计算OOB得分。此外，我们希望使用的验证集是最近的日期，而不是随机的。

```python
def print_score(m):
    res = [rmse(m.predict(trn), y),
           rmse(m.predict(val), y_val),
           m.score(trn, y), m.score(val, y_val)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222204825152.png)
均方根误差为0.73。
接下来调参```min_samples_leaf=10```:
```python
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=10, 
                          n_jobs=8)
%time m.fit(x, y)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222205547491.png)
得分下降为0.6。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222210227730.png)
接下来继续调参```min_samples_leaf=3```:
```python
m = RandomForestRegressor(n_estimators=20, min_samples_leaf=3, 
                          n_jobs=8)
%time m.fit(x, y)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018122220555812.png)
得分下降了很多，只有0.58了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222205809685.png)

得分实际上不太好。让我们回过头来再看看训练用的数据集。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222173632972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

这些是我们必须预测的列(加上``add_datepart``添加的列)。大部分关于明天卖多少东西的见解，可能都包含在商店在哪里，他们在商店里想卖什么东西，对于一个给定的商品，它是什么种类的商品等细节中。随机森林除了在星期几、商店号、商品号等项目上创建二元分割之外，没有其他功能。它不知道商品的类型或商店的位置。由于它理解当前情况的能力有限，我们可能需要使用整个4年的数据来获得一些有用的见解。但是一旦我们开始使用整个4年的数据，我们使用的很多数据都是很古老的。有一个Kaggle kernel给出了一个建议：

 1. 以过去两周为例。
 2. 计算商店号、商品号、促销活动分组的平均销售额，然后用整个日期的平均值。
 3. 提交这个值就能排名30名左右。
我们将在下节课讨论这个，但是如果你改进那个模型，你将会排在第30位以上。
>能否通过创建新的列来捕捉季节性和趋势效应，比如8月份的平均销售额(average sales) ?
>这是个好主意。要解决的问题是如何去做，因为有很多细节需要处理，而且很难——不是智力上的困难，但它们的难度会让你在凌晨2点撞到桌子。

>给机器学习撸码是非常令人沮丧和困难的。如果您弄错了某个细节，大多数情况下它不会报一个异常，只会默默地比其他情况稍微差一些。如果你在Kaggle上，你就会知道你的表现不如其他人。但除此之外，你没有什么可与之相比的。现实中，你不知道你的公司的模型是属于好的那一半，还是不好的那一半，仅仅因为你犯了一个小错误。这也就是在Kaggle上练习很爽的原因。
你会惊讶的发现，你能在练习时找出所有可能把事情搞砸的方法。

>你经常从其他来源获取数据来补充你现有的数据集么?
>没错。星型模式的全部意义在于您拥有一个中央表，并从中派生出其他表来提供关于它的元数据。在Kaggle上，大多数竞赛都有这样一条规则:你可以使用外部数据，但是你得在论坛上发帖公开数据(请仔细检查规则!)现实中，你应该始终寻找并尽可能利用的外部数据。

>有另一个kaggle竞赛和这个基本一样。获胜的人是从事物流预测的领域专家和专家。他根据自己的经验创建了很多列，这些经验有助于做出预测。第三名的获得者几乎没有做任何特征工程，不过他们也有一个大的疏忽，而这个可能使他们错失了第一名。之后我们将学习更多关于如何赢得这种类型的比赛。

### 1.4 好验证集的重要性
如果没有一个好的验证集，就很难创建一个好的模型。如果你建立模型想预测下个月的销售，可是你无法知道构建的模型是否能够提前一个月预测销售；那么当模型投入生产环境时，你也无法知道它是否真的有效。所以你需要一个知道可靠的验证集，它可以告诉你，当模型投入生产或在测试集中使用时，它是否能够很好地工作。
通常情况下，测试集用于在比赛结束时或者在项目结束时测试模型做得如何。但是还可以使用测试集校准验证集。

这里举一个例子。Terrance做了四个不同的模型，并且提交给了Kaggle评分。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222213657689.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

x轴是Kaggle在排行榜上告诉我们的分数，y轴是他在一个特定的验证集上得到的分数，他想看看这个验证集是否**特别**有用。如果你的验证集是很好，那么排行榜分数(即测试集分数)之间的关系应该是一条直线。理想情况下，应该位于y = x线上，但这并不重要，只要它相对地告诉你哪个模型比别的更好，然后你就知道哪个模型是最好的。在这种情况下，Terrance成功地提出了一个验证集，它看起来可以很好地预测Kaggle排行榜的得分。这真的很酷，因为他可以去尝试100种不同类型的模型，特性工程，加权，调整，超参数，等等，看看他们如何进行验证集，而不必提交给Kaggle。这不仅适用于Kaggle，也适用于您所做的每一个机器学习项目。一般来说，如果验证集没有显示出很好的匹配线，就需要好好想想了。测试集是如何构造的?我的验证集有何不同?你需要画很多图表以寻找答案。
>如何构造一个与测试集非常接近的验证集?
>以下是Terrance的一些建议:
> - 按截止日期(即最近日期)
> - 首先查看测试集的日期范围(16天)，然后查看kernel的日期范围，它描述了如何平均(14天)在排行榜上获得0.58。
> - 测试集开始于发薪日之后，结束于发薪日。
> - 多画图。即使不知道那天是发薪日，你也可以绘制时间序列图，并观察每两周出现一次峰值，并确保验证集中的峰值数量与测试集中的峰值数量相同。

## 2 解析机器学习模型
接下来回到之前blue book for bulldozers的竞赛中来。
```python
set_plot_sizes(12,14,16)
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
def split_vals(a,n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)
```
设置评分函数。
```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
```
### 2.1 分析数据
一旦我们完成``proc_df``，这就是它的样子。SalePrice是销售价格的对数值。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223200914466.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)我们已经知道如何进行预测。在每棵树中运行特定行之后，我们取每棵树中每个叶节点的平均值。通常，我们不仅想要一个预测——我们还想知道我们对这个预测有多自信。

如果我们没有看到很多像这样的行示例，预测的置信度就会降低。在这种情况下，我们不会期望任何树都有一条路径来帮助我们预测这一行。所以从概念上讲，当你把这个特定行传给不同的树，它会在不同的地方结束。换句话说，如果不只是取树的预测的均值作为预测结果，而是取预测值的标准差呢?如果标准差预测差很高，这意味着每棵树对这一行给出了非常不同的估计，在使用这些结果时应该更加谨慎。如果这是一种非常常见的行，树就会学会对它做出很好的预测，因为它已经看到了许多基于这种行进行分割的机会。因此，树间预测的标准预测差至少让我们相对了解了我们对这个预测有多自信。这不是scikit-learn中存在的东西，所以我们必须创建它。不过我们差不多已经有所需的代码了。

对于模型解析，不需要使用完整的数据集，因为我们不需要非常精确的随机森林——我们只需要一个表明所涉及的关系的性质的森林。

只要确保样本大小足够大，那么多次调用相同的解释命令，就不会每次得到不同的结果。在实践中，50,000是一个很大的数字了，如果这还不够的话(它以秒为单位运行)。
```python
set_rf_samples(50000)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 
                        max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223202236727.png)
```python
%time preds = np.stack([t.predict(X_valid) for t in m.estimators_])
np.mean(preds[:,0]), np.std(preds[:,0])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223202829230.png)
这是一种观察的方法。这需要相当长的一段时间，特别是它没有利用电脑有的多核心。如果Python代码以串行方式运行，则列表理解本身就是串行的，这意味着它在单个CPU上运行，不会利用多CPU硬件。如果我们想在更多的树和数据上运行它，执行时间就会增加。Wall时间(实际花费的时间)大致等于CPU时间，如果它在许多内核上运行，那么CPU时间将高于Wall时间。

当我们使用python像这样循环遍历树时，我们是按顺序数计算每个树的，这是很慢的!我们可以使用``parallel_trees``并行处理来加速:
```python
def get_preds(t): return t.predict(X_valid)
%time preds = np.stack(parallel_trees(m, get_preds))
np.mean(preds[:,0]), np.std(preds[:,0])
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223203413694.png)

 - ``parallel_trees``接收一个随机森林模型``m``和要调用的函数(这里是``get_preds``)。它并行地在每棵树上调用这个函数。
- 它将返回将该函数应用于每棵树的结果列表。
- 这将缩短wall时间，并给出完全相同的答案。
### 2.2 画图分析
你们可能还记得上节课我们讲过的一个预测符叫做``Enclosure ``。让我们从直方图开始。

```python
x = raw_valid.copy()
x['pred_std'] = np.std(preds, axis=0)
x['pred'] = np.mean(preds, axis=0)
x.Enclosure.value_counts().plot.barh();
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223204322715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
> pandas的优点之一是它有内置的[绘图功能](https://pandas.pydata.org/pandas-docs/stable/visualization.html)。

> ``Enclosure``是个啥?不知道，其实这无关紧要。这个过程的目的是让我们了解哪些事物是重要的，然后找出它们是什么以及它们有多重要。所以我们一开始对这个数据集一无所知。我们要看的是一种叫做``Enclosure``的东西，它有一种叫做``EROPS``和``ROPS``的东西我们还不知道是什么。我们所知道的是，大量出现的三个变量是``OROPS``、``EROPS w AC``和``EROPS``。
> 这对于数据科学家来说是很常见的。你经常发现自己在查看不太熟悉的数据，你必须谨慎地研究哪些位需要仔细研究，哪些位似乎很重要，等等。在这种情况下，至少要知道``EROPS AC``，``NO ROPS``，``None or Unspecified``几乎没有量，我们可以不关心这几个东西，而要重点关注``OROPS``, ``EROPS w AC``，和``EROPS``。
在这里，我们将data frame按``Enclosure``分组，然后取3个字段的平均值：
```python
flds = ['Enclosure', 'SalePrice', 'pred', 'pred_std']
enc_summ = x[flds].groupby('Enclosure', as_index=False).mean()
enc_summ
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223205521791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)我们可以从这里发现一些东西:
- 预测与售价接近(好现象)
- 标准差有一点变化
```python
enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018122320584740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
```python
enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', alpha=0.6, xlim=(0,11));
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018122320593042.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)上面的误差条的使用了预测标准差。这将告诉我们是否某些组或行的置信度是否够好。
我们可以对``product size``也这么干:
```python
raw_valid.ProductSize.value_counts().plot.barh();
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223210741937.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
```python
flds = ['ProductSize', 'SalePrice', 'pred', 'pred_std']
summ = x[flds].groupby(flds[0]).mean()
summ
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223211004348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
平均情况下，当预测一个更大的数字时标准差会更高。可以根据预测的标准差和预测本身的比值排序。
```python
(summ.pred_std/summ.pred).sort_values(ascending=False)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223211119167.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
这告诉我们，对于``product size``中``Large`` 和``Compact``，我们的预测是不准确的(相对来说，作为总价格的比例)。这些是直方图中最小的组，在预测小的组时我们做得不好。

这个置信区间主要用于两个目的:

 1. 你可以分组查看平均置信区间，看看我们是否对某些组不够自信。
 2. 也许更重要的是，你可以查看它们的特定行。当你把它投入生产时，你可能总想看到置信区间。例如，如果你在做信用评分来决定是否给某人贷款，你可能不仅想知道他们的风险水平，还想知道我们有多自信。如果他们想借很多钱，而我们对预测他们是否会还款的能力一点信心都没有，我们可能会给他们一笔较小的贷款。

## 3 特征重要度
> 仅仅知道一个模型能够做出准确的预测是不够的——我们还想知道它是如何做出预测的。其最重要的方法是特性重要度。
> 在实践中，我总是首先考虑特性的重要性。无论我是在Kaggle竞赛还是在真实项目中工作，我都会尽可能快地构建一个随机森林，试图让它比随机好得多，但不是必须要好得多（？？）。然后就是画出特征重要度。
### 3.1 使用特征重要度
特征重要度可以告诉我们，在这个随机森林中哪些列是重要的。
```python
fi = rf_feat_importance(m, df_trn); fi[:10]
```
在这个数据集中有很多列，我们选出了前10名最重要的。``rf_feat_importance``来自Fast.ai库，接收模型``m``和dataframe ``df_trn``(因为我们需要知道列的名称)，返回一个panda dataframe，按照重要性的顺序显示每个列的重要性。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223213612409.png)
```python
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
```
由于``fi``是一个DataFrame，可以使用``DataFrame``绘图命令。重要的是，要看到有些列非常重要，而大多数列并不重要。在现实生活中使用的几乎所有数据集中，这差不多就是特性重要度的套路。需要你关注的列屈指可数。在这一点上，就研究重工业设备拍卖这一领域而言，我们只需要关心那些重要的列。我们要学习``Enclosure``吗?取决于``Enclosure``是否重要。实际上它出现在前10名里了，所以我们要研究一下。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223212738380.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)我们也可以把它画成条形图:
```python
def plot_fi(fi): 
  return fi.plot('cols','imp','barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181223213830772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
现在最重要的事情就是逐个分析这些高重要度的属性，比如画``YearMade``对比价格的直方图或者散点图，从中学习所有你能学习的东西。
在实际项目中经常发生的情况是，您对客户说：“原来Coupler_System是第二重要的东西。”但是他们可能会说：“这个属性没有意义”。这并不意味着你的模型出了问题，而是说明客户对他们提供给你的数据的理解出了问题。
>Jeremy举了个例子：我参加了一次Kaggle竞赛，目的是预测申请哪所大学的助学金会成功。我使用了这种方法然后发现了一些列几乎完全可以预测因变量。具体地说，当我观察它们在哪些方面具有预测性时，我发现它们的值是否丢失是这个数据集中唯一重要的事情。多亏了这种洞察力，我最终赢得了那场比赛。后来，我听说发生了什么事。事实证明，在那所大学填写任何其他数据库都是一种管理负担，所以对于许多拨款申请，他们不为那些申请未被接受的人填写数据库。换句话说，数据集中缺少的这些值表示不接受拨款，因为如果被接受，管理员就会输入该信息。这就是我们所说的**数据泄漏**。数据泄漏意味着我所建立的数据集中有一些信息在现实生活中大学做决定的时是不会有的。当他们真正决定要优先考虑哪些拨款申请时，他们不知道管理人员稍后会向哪些申请添加信息，因为他们最终被接受了。

``Coupler_System``告诉您某一特定类型的重工业设备是否具有特定的特性。但如果该设备根本不是那种工业设备，``Coupler_System``值就会缺失。由此可见，``Coupler_System``可以表明设备是否属于某一类重工业设备。这不是数据泄漏，而是你得到的实际信息。你只是需要谨慎地解释它。所以你至少应该浏览一下前10名（或者截至到多少名），然后仔细研究这些属性。
为了方便，有时候最好扔掉一些数据，看看是否有什么不同。我们过滤掉那些重要性等于或小于0.005的部分(即只保留重要性大于0.005的部分)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181224215600648.png)
r²变化不大，实际上增加了一点。一般来说，删除冗余列不会使情况变得更糟。如果这让情况变得更糟，说明它们不是多余的。这可能会让得分更好一点，因为如果想想这些树是如何构建的，当它决定要分割什么时，它不用担心尝试什么，它很少会意外地找到一个糟糕的列。因此，用更少的数据有更大的几率去创建更好的树（不过不会有太大的改变）。但这会让它更快一点，让我们把注意力集中在重要的事情上。
再次运行这个新结果的特性重要度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181224220022917.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
当删除冗余列时，也删除了共线性的来源。换句话说，这两列可能是相互关联的。共线性并不能使你的随机森林预测较少,但如果你有一个列A和列B有点相关，B是一个强大的独立变量，那么重要度将从A和B之间分裂。通过删除一些影响很小的列，能使得特征重要度图像更清晰。之前的``YearMade``离``Coupler_System``很近，但是肯定有很多东西和``YearMade``是共线的，所以删掉后，重要度排列改变了，可以看到``YearMade``真的很重要。这个特征重要度图比之前的更可靠，因为它的共线性更小，不容易误导我们。
### 3.2 特征重要度的工作原理
>这个特征重要度不仅非常简单，而且是可以应用在随机森林或者其他任何机器学习模型中的一种技术（有趣的是，几乎没有人知道这一点。许多人会告诉你没有办法解释这一特定种类的模型(模型的最重要的解释是知道哪些东西是重要的)，Jeremy说他们错了，因为他教你的这个技术适用于任何类型的模型）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181224221400638.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
下面是这项技术涉及的步骤：
- 对于这个推土机的数据集，我们试图预测列``Price``(因变量)。
- 有25个自变量，其中一个是``YearMade``。
- 怎么知道``YearMade``有多重要?我们有一个完整的随机森林可以得到预测精度。把所有行放到随机森林中，得到一些预测，然后用它们和实际价格比较(在本例中，使用的是的均方根误差以及r²)。这是我们的起点。
- 接着做完全相同的事情，但这次取``YearMade``列并随机洗牌(即只对该列进行随机排列)。现在``YearMade``的分布与以前完全相同(相同的均值，相同的标准差)。但它和因变量完全没有关系因为我们完全随机地重新排序。
- 之前的r²是0.89。洗牌``YearMade``之后r²是0.80。所以破坏这个变量后，分数变得更糟了，得分减少了0.09。
- 接下来回到洗牌之前，这一次对``Enclosure``洗牌。这次r²是0.84。得分减少了0. 05。说明``Enclosure``没有``YearMade``重要。由此可以得出每个列的特征重要度。

>关于特征重要度技术，移除列并训练一个全新的随机森林，然后算出得分吗?那将会非常慢。相比之下，我们只需要为每个打乱的列重新运行林中的每一行，这很好也很快。

>如果你想知道哪对变量是最重要的，你可以依次做同样的事情，挑一对变量洗牌。显然是计算上非常昂贵的，在实践中，有更好的方法来做到这一点。


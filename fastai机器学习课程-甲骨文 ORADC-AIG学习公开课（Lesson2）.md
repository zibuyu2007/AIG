>Hi，这是我们第2次公开课。之所以有这个分享课程是因为大家太忙（懒），没有时间看fastai在线视频和笔记。而且视频和笔记都是英文的，大家也不想费脑子（懒）。所以本课程的目的就是把Jeremy老师的视频用中文再给大家讲一遍并且上传到国内的视频网站，另外把Hiromi小姐的笔记翻译加工一下分享给大家。
--------------------- 

# 深入随机森林
### [Notebook](https://github.com/fastai/fastai/blob/master/courses/ml1/lesson1-rf.ipynb)
## 1.评估方法
### 1.1 RMSE定义
上回讲到调用了一个基础模型，把train数据集传入并打印了结果评分。
```python
def rmse(x,y): return math.sqrt(((x-y)**2).mean())
```
这里用到了df的[mean()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mean.html)函数，用于返回平均值。另外，由于我们已经对传入的x,y取对数了，
```python
df_raw.SalePrice = np.log(df_raw.SalePrice)
```
所以这里只需要计算rmse就可以啦。
>%time可以记录cpu运行的时间。根据经验来说，一般运行时间超过10秒，就无法进行互动分析了。所以在调研阶段要控制其在一个合理的运行时间。而等到下班前把当天的特征工程、参数都做得差不多了，就可以跑一长时间的试试，第二天回来再看看结果。

```python
def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)
```
> 均方根对数误差，Root Mean Squared Logarithmic Error (RMSLE):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181203153523456.png)
均方根误差，Root Mean Squared Error (RMSE):
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181203153633144.png)

然后我们把数据集里所有的数据都转换成数字：

 - ```add_datepart ```—获取日期特征。
 - ```train_cats```—转换string为pandas的category数据类型。然后调用```proc_df```把所有category类型的列，都替换为其category codes值。
 - ```proc_df```还可以把缺失值替换为中位数，添加[column_name]_na的列，然后通过设置其值（true/false）表示该列是否缺失值。
 ### 1.2 R²定义
**R²是什么？
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181203155300134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
> yi是目标数据，ȳ 是平均值，SStot是数据的差异，fi是预测数据，SSres是实际模型的RMSE。
>如果预测结果和平均值完全一样（fi=ȳ ），则SSres/SStot=1, R² = 0。
>如果预测结果和实际值完全一样（fi=yi），则SSres/SStot=0, R² = 1。
>其中R²的范围是1~-∞。如果值是负的，说明你的模型太差，全预测平均数也比你的模型好。

>你不需要对R²进行调优，它更多是提供一个模型好坏的趋势参考。比如R²是0.8的时候模型是什么样的，R²是0.9的时候模型是什么样的。有时候可以用随机噪音数不同的数据集，创建一个2D散点图，看一下R²的值，感受一下这些和实际值的接近程度。
>R²是衡量你的模型与全平均值~~模型~~ 的比率。总要比无脑全预测平均值要好才行啊。
https://www.graphpad.com/guides/prism/7/curve-fitting/index.htm?r2_ameasureofgoodness_of_fitoflinearregression.htm

### 1.3 过拟合（Overfitting）
**验证集和测试集的区别？**
>如果只有一个数据集，然后在其上尝试了各种超参数的组合，最后模型可能会在这个数据集上过拟合。所以如果有一个验证集，我们就可以在上面验证模型是否合适。如果结果在训练集上很精确，但是在验证集上很糟糕，说明模型过拟合了，需要调整超参数。
测试集则只是用于测试最终模型的好坏的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181203173020333.png)
可以看到，R²在训练集上是0.982，在验证机是只有0.886，说明我们的模型过拟合了，但是RMSE是0.25，所以过拟合的不是特别厉害。

**为什么不随机选择数据生成验证集呢？**
>因为数据集里的日期是一个重要数据，如果任意选择一些日期组成验证集，实际上对于预测的计算来说就变得更容易了，因为我们要预测后面日期的数据，在训练集里已经出现了差不多日期的数据了，验证集里日期相近的数据很容易就能预测出来。但是对于测试集，用的是单独的日期期间的数据，预测结果就不会太好了。结论就是，通常如果建立的模型包含日期元素，测试集又是单独的日期期间的话，相应的验证集也应该使用单独的日期期间的数据。

**最终模型会不会在验证集上也过拟合呢？**
>会的。随着一次次提交给kaggle，模型可能会在验证集和测试集上过拟合（一般进行中的比赛，都是最后几天才放出测试集，在测试集上的结果决定了public leader board的排名），但是kaggle有一个private leader board set，使用的数据集不会提供出来，所以在private leader board的排名更可以彰显实力。这说明要真正会调参，创建一个好模型才是关键。

## 2 随机森林
### 2.1 加速训练

为了加速运行，调研阶段我们可以用数据子集来训练。
```python
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000) 
y_train, _ = split_vals(y_trn, 20000)
```
这次训练只用了不到3秒。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181207164232227.png)
有两点需要注意：
- 要确保验证集没有改变（使用"_"接收并抛弃返回的另一部分df）。
- 要确保各个训练子集的日期没有重叠。

 ### 2.2 创建一棵树
 
 ```python
 m = RandomForestRegressor(n_estimators=1, max_depth=3,
                          bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
 ```
 - ```n_estimators=1```——estimators 默认是10，是树的数量。
 - ```max_depth=3```——只分裂3层的小树苗
 - ```bootstrap=False```——关闭随机森立的随机化功能
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181207165231805.png)

这颗小决策树的 R²只有不到0.4，是不是个好模型，但是比都预测为平均值要好点。
```python
draw_tree(m.estimators_[0], df_trn, precision=3)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181207165554718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
这棵树由一连串的二元决策节点组成。
- 每个节点的第一行是用于决策的条件
- 第一个节点中samples=20000是初始记录数
- 深颜色的节点表明value比较大
- value是price的log平均数，如果我们建一个全都是取平均值的模型，那么mse就是0.495。
- 最优的分裂条件，在第一个节点中是Coupler_system ≤ 0.5。这个分裂条件，是取一个变量的值，能够使分裂后的两组越不同越好。我们要尝试所有变量的所有值，以判断是不是分裂条件够好。而判断的一句就是生成两个新节点的MSE加权平均数小于原始节点的。
- 停止分裂的条件，一是到达了max_depth最大深度，一是叶子节点中只剩一个column了。
### 2.3 创建一个更深的树
```python
m = RandomForestRegressor(n_estimators=1, bootstrap=False, 
                          n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181207172920930.png)
把最大深度去掉以后，在训练集上的 R²变为1了，因为我们真正走到最后，每个叶节点都只有一个元素了。验证集上是0.57，比刚才0.39好点了。
## 3 Bagging
### 3.1 bagging介绍
Michael Jordan 发明了一种技术Bagging（Bootstrap AGGregatING，**引导聚合**），可以应用于任何模型并使其更健壮。随机森林正是应用了这种技术，去bagging树。
>[bagging是什么？](https://blog.csdn.net/u012151283/article/details/78104678)
>如果我们创建了五个不同的模型，每个模型只是在某种程度上具有预测性，但模型给出了彼此不相关的预测。 这意味着这五个模型将对数据中的关系有深刻的不同见解。 如果你采用这五种模型的平均值，就可以有效地从每种模型中获取洞察力。 所以这种平均模型的想法是一种聚合技术。
<br>
如果我们创建了很多大而深、大量过度拟合的树，对每一棵树我们只随机选择1/10数据，然后重复这样做了一百次（每次都是不同的随机样本）。虽然它们过度拟合，但由于它们都使用不同的随机样本，所以它们是在不同的东西上以不同的方式过度拟合。 换句话说，它们都有错误，但错误是随机的。 一堆随机错误的平均值为零。 如果我们取这些树的平均值，每个树都经过不同的随机子集训练，那么误差将平均为零，剩下的就是真正的关系——也就是 [随机森林](https://baike.baidu.com/item/%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97/1974765?fr=aladdin)。

>即便我们创建10个糟糕的模型，然后对其取平均值，只要它们都是基于不同的子集，而且错误都是不相关的，我们也可以得到一个不错的预测结果。根据scikit-learn的处理实现方式，这些子集有可能是会重叠的。在实践中，使用随机森林空间来找到最近的邻居与使用 [欧几里得空间](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%B7%E7%A9%BA%E9%97%B4/8281167?fr=aladdin) 之间的区别就在于，做出好的预测的模型与做出无意义预测的模型之间的区别。

>机器学习建模的唯一目的是找到一个模型，它能告诉你哪些变量是重要的，以及它们如何相互作用来驱动因变量的。有效的机器学习模型能够准确地找到训练数据中的关系，并能很好地推广到新的数据。
在bagging中，这意味着每一个单独的估计值，你希望它们尽可能具有预测性，但是对于你的单独树的预测要尽可能不相关。研究团体发现，更重要的似乎是**创建不相关的树**，而不是更准确的树。
在scikit-learn中，还有一个类叫做```ExtraTreeClassifier```，它是一个非常随机的树模型。它不是尝试每一个变量的分割，而是随机地尝试几个变量的分割，这使得训练更快，它可以建立更多的树——更好的泛化。如果你有一个糟糕的模型，你只需要更多的树来得到一个好的结束模型。

### 3.2 实践bagging
接下来再次执行基本模型。
```python
m = RandomForestRegressor(n_jobs=-1) 
m.fit(X_train, y_train) 
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208113308942.png)
然后对每一棵树进行单独预测，再取平均值。
```python
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208113631660.png)
```python
preds.shape
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208113926373.png)
每个树都存储在一个名为```estimators_```的属性中。我们对每棵树的验证集都调用predict。```np.stack```将它们连接在一个新的轴上，因此预测结果的形状为(10,12000)(10棵树，12,000个验证集)。第一个数据的10个预测的平均值是9.51，实际值是9.10。可以看到，每个单独的预测都不接近9.10，但是它们平均值却非常接近9.10。

Here is a plot of R² values given first i trees. As we add more trees, R² improves. But it seems as though it has flattened out.
基于前i个树的R²值，我们绘制一个plot折线图。随着我们添加更多的树，R²也随之改善，但它将会越来越趋于平缓，也就是说越来越难以改进了。
```python
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208114738878.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)
接下来我们添加更多树以验证这个结论。
```python
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208115106928.png)
```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208115116443.png)
```python
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208115129221.png)
>添加更多的树会减慢计算速度，但是使用更少的树仍然可以得到相似的预测结果。所以当Jeremy建立他的大部分模型时，他从20到30棵树开始，在项目结束或一天工作结束时，他会用1000棵树，然后通宵运行，第二天再看看结果。

这是第一个要学习设置的**超参数**——一堆**estimators**。一种设置原则是，只要你有足够的时间去fit，而且结果是有改善的，那就去多设置试试。

### 3.3 Out-of-bag (OOB) 评分
有时你的数据集很小，您不想提取验证集，因为这样做意味着你没有足够的数据来训练一个好的模型。然而，随机森林有一个非常聪明的技巧，称为[out- bag error（包外估计错误）](http://blog.sina.com.cn/s/blog_4c9dc2a10102vl24.html)，它可以处理这种情况（以及其他更多情况）。

[其思想是](https://wiki.hyper.ai/article/2159/)：由于基分类器是构建在训练样本的自助抽样集上的，只有约 63.2％ 原样本集出现在中，而剩余的 36.8％ 的数据作为包外数据，可以用于基分类器的验证集。

具体来说：我们的第一棵树中，一些行没有被用于训练，将那些未使用的行传递到第一棵树，并将其视为一个验证集。对于第二棵树，我们可以传递未用于第二棵树的行，依此类推。

实际上，我们会为每棵树设置不同的验证集。为了计算我们的预测，我们需要对所有不使用这一行进行训练的树进行平均。如果有数百棵树，那么很可能所有的行都将多次出现在这些out-of-bag示例中。你可以计算RMSE、R²等这些out-of-bag预测。

>这还有一个好处，即允许我们查看模型是否泛化，即便我们只有少量的数据而不想分出单独的验证集。

只需向模型构造函数中添加一个参数```oob_score```即可，print_score函数将把OOB error打印在最后的位置。
```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208121644400.png)
结果表明，我们的验证集日期差异造成了影响，模型过拟合了。

>我们希望自动设置一些超参数。Scikit-learn有一个名为grid search的函数，可以接收一个列表，其中包含你想要优化的所有超参数以及这些超参数的所有值。它将在所有这些超参数的所有可能组合上运行模型，并告诉你哪一个组合是最好的。

## 4 减少过拟合
### 4.1 二次抽样
事实证明，避免过拟合的最简单方法之一也是加速分析的最佳方法之一：二次抽样。让我们返回到使用完整数据集，以便演示这种技术的影响。
```python
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
set_rf_samples(20000)
```
基本思路是这样的：与其限制模型可以访问的数据总量，不如将其限制为每个树的*不同*随机子集。这样，在给定足够多的树的情况下，模型仍然可以看到所有的数据，但是对于每一个单独的树来说，它的速度就和我们之前减少数据集的速度一样快。
```python
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208142820790.png)
由于每添加一棵树允许模型查看更多的数据，因此这种方法可以使添加树更有效。
```python
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208142832521.png)运行时间与之前差不多，但是每棵树都可以访问整个数据集。使用40棵树后，在验证集上的R²为0.877。
### 4.2 建树的其他参数
#### 4.2.1 基准值
为了显示其他过拟合避免方法的影响，我们恢复使用完整的引导样本。
```python
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208144647118.png)
这里OOB的R²得分要比验证集高，这是因为我们的验证集是一个不同的时间段，而OOB样本是随机的。预测一个不同的时间段要困难得多。
```python
t=m.estimators_[0].tree_
def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else: # leaf
            return 1

    root_node_id = 0
    return walk(root_node_id)
dectree_max_depth(t)
```
树的最大深度是45，以及验证集R²的0.897将作为基准。

#### 4.2.2 min_sample
另一种减少过拟合的方法是减少树的深度。我们通过指定(使用min_samples_leaf)每个叶子节点中需要一些最少的行数来实现这一点。这有两个好处：
- 每个叶节点的决策规则较少，更简单的模型应该更好地泛化。
- 这些预测是通过平均叶节点中的更多行来实现的，从而减少了波动性。

```python
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018120814495927.png)
```python
t=m.estimators_[0].tree_
dectree_max_depth(t)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/201812081451521.png)
使用min_samples_leaf=5之后（默认是1），验证集的R² 从0.89 提高到了0.90。

``min_samples_leaf`` : 叶结点需要最少的样本数，也就是最后到叶结点，需要多少个样本才能算一个叶结点。如果设置为1，哪怕这个类别只有1个样本，决策树也会构建出来。如果min_samples_leaf是整数，那么min_samples_leaf作为最小的样本数。如果是浮点数，那么min_samples_leaf就是一个百分比，同上，celi(min_samples_leaf * n_samples)，数是向上取整的。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

继续减小min_samples_leaf的值，结果发生了过拟合。
```python
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208145558879.png)
#### 4.2.3 max_features
我们还可以增加树之间的变化量，不仅可以为每棵树使用行样本，还可以为每一次分裂使用列样本。我们通过指定``max_features``来实现这一点，``max_features``是在每次分裂时随机选择的特性的比例。
```python
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208153028748.png)
-  ``max_features=0.5``：这个想法是树之间的相关性越小越好。试想如果你有一列比所有其他列都要好的多，因为它具有预测性所以，你构建的每棵树都是从这一列开始的。但是可能存在一些变量的相互作用，这些相互作用比单个列更重要。所以如果每棵树第一次总是在同一个地方分裂，这些树就不会有太大的变化。
- 除了取行子集之外，在每一个拆分点上，还取列的不同子集。
- 对于行抽样，每棵新树都是基于一组随机的行；对于列抽样的每一个单独的二元分裂，我们从不同的列子集中选择。
- 0.5表示随机选择其中的一半。你也可以使用一些特殊的值，如sqrt或log2。最好的值是1、0.5、log2或sqrt。

sklearn docs 提供了使用不同``max_features ``并增加树的数量的[例子](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)——可以看到，在每次分裂上使用一部分特征要求更多的树，但是生成了更好的模型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181208153210477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzIwMjE2NDU1,size_16,color_FFFFFF,t_70)

下一节课，我们将学习如何分析模型，以便更好地了解数据。

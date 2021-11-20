# GML-DM-hw1
Graph machine learning and data mining, homework1

成员：江彦泽、俞越、陈励

该README最后会作为报告提交，报告中包括：
- 详细的实验过程
- 遇到的问题和解决方案
- 最后得到的分析性数据结果或结论

环境信息：
- 操作系统：Windows 10
- Python: 3.8.8
- PyTorch: 1.10.0 py3.8_cpu_0
- PyG: 2.0.2 py38_torch_1.10.0_cpu

## 任务一 网络数据分析
对比分析给定的三个不同数据集的性质
- 图的平均节点度数；
- 画出度分布直方图，横轴k代表度的取值，纵轴P(k)代表任取结点度数为k的概率；
- 图的平均节点聚集系数；

### 1.1 读取数据
> 详细代码见：degree_analysis.py 的 read_dataset_from_file 函数

我们使用PyG完成任务，因此，需要将数据读入torch_geometric.data.Data中 

首先，我们读取edge_list.txt文件，读取边的信息 "edge_index"。

然后，我们需要读取feature.txt文件，将node的特征信息 "x" 读入dataset，这个信息虽然不会在任务一中用到，但之后的任务中可能需要使用。

最后，我们将edge的信息 "edge_index" 和节点的feature信息 "x"作为参数传入torch_geometric.data.Data，构造出待使用的数据集。

### 1.2 图的平均节点度数
> 详细代码见：degree_analysis.py 的 cal_avg_degree 函数

设图的边数为E，节点数为V，则：
- 无向图平均度数计算公式为：2*E / V
- 有向图的平均度数公式为：E / V

这次任务给的数据都是无向图，但写代码的时候考虑了有向图和无向图2种情况。

degree_analysis.py 的 main 运行结果如下:
```
dataset/cora/'s average degree is 7.796159527326441
dataset/chameleon/'s average degree is 55.10935441370224
dataset/actor/'s average degree is 14.031052631578948
```

### 1.3 画出度分布直方图
> 详细代码见 degree_analysis.py 的 draw_degree_distribution_histogram 函数

其思路为，先算出每个点的度数，然后统计每个度数出现的次数，最后画出频率分布直方图。最终得到的度分布直方图如下：
![degree_distribution_histogram](images/1.3.png)
上面一行`original`指的是不做任何处理的概率分布直方图，但由于一些边缘数据取到的概率本就不大，因此做了一个展示上的优化，
取P > 0.01的数据画了第2行图。现在代码的运行结果就是第2行图。
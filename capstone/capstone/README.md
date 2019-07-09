## 文件内容含义

- datasets
    目录，存放数据集

- models
    目录，存放训练好的模型

- results
    目录，存放测试集预测结果

- solutions
    目录，项目解决方案相关文件
    - base_line.ipynb 
        使用logistic regression创建基线模型

    - data_visualization.ipynb
        对训练集进行可视化分析

    - `preprocess.py`

        数据预处理代码，实现函数如下：  
        - extra_address_for_block
            解析Address字段是否含有Block字段，在原数据集中添加一列数据

        - extra_dates
            解析Dates字段中的年、月、日、时，在原数据集中添加四列数据

        - extra_address_for_suffix
            解析Address字段中的道路类型(后缀)，在原数据集中添加一列数据

        - extra_address_for_infos
            解析Address字段中的道路名称、道路编号，在原数据集中添加三列数据

        - dataset_sample
            在DataFrame中随机取样且保证每个Label都有样本

    - solution.ipynb

        项目的解决方案流程及实现。
        由于训练时间超级长，因此很多训练过程都是拆成多个文件实现的，最后拼接到一个文件。中间执行结果还在，但是不要轻易执行该文件，会跑几天的时间。

        提供直接使用保存下来的模型进行复现。

    - offline_model.ipynb
        直接运行后，使用保存在本地的模型对测试集进行预测并保存指定格式的结果文件。


## 离线文件下载路径
    链接: https://pan.baidu.com/s/1LyT1OuXTYx4F09sX-hg8UQ 提取码: smc3


## 代码运行时间

运行环境信息：
![df](./systeminfo.png)

整个solution.ipynb跑下来可能需要几天时间，所以不建议本地跑一边

百度云上保存了保存到本地的模型，下载后可以直接使用offline_model.ipynb文件跑一边，使用整个树预测整个测试集的话，大概需要5个小时。

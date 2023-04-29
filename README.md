目录结构

```
Project
├── checkpoints 保存模型参数和日志记录
    ├── minist_classifier # 使用模型名字，来区分不同模型的参数
        ├── name1 # 这一层使用不同训练参数命名，来区分相同模型结构，不同训练超参的参数
            ├── log    # tensorboard 保存数据的目录
            ├── best_*.pkl    # 验证集上最优的模型参数
            ├── fina_*.pkl    # 最后的模型参数
            ├── log.txt        # 运行过程中的日志文件
        ├── name2
    ├── model2
├── data    # 存放数据集及数据处理相关代码
    ├── minist    # minist数据集
        ├── raw    # 原始数据 
        ├── train   # 训练集 
        ├── test    # 测试集
        ├── val    # 验证机
    ├── argument.py    # 数据增广操作
    ├── minist_dataset.py    # minist 数据集的Dataset实现
    ├── preprocessing.py    # 数据预处理，比如讲原始数据划分为train/test/val等
├── model
    ├── basemodel.py    # 所有模型的及模型
    ├── minist_classifier.py   #minist分类器模型
    ├── operators.py    # 实现自定义操作，或者公有操作
├── script        # 训练shell脚本目录
    ├── train.sh    
├── server    # 存放服务器相关文件
├── sota    # 存放对比方法代码
    ├── sota1
    ├── sota2
├── util    # 工具类
    ├── logger.py    # 日志记录相关
    ├── setup.py    # 环境设置相关，比如随机种子
    ├── tools.py   
├── yaml
    ├── config.yaml    # 通过yaml更新Argsparser
├── eval.py    # 验证模型性能的，比如论文中相关量化数据和可视化数据
├── loss.py    # 自定义loss函数
├── main.py    # 程序入口
├── metrics.py    # 自定义metrics的类    
├── options.py    # 设置程序的参数
├── trainer.py    # 训练入口
```

# data

1. 该目录存放整个项目的数据机器相关的操作，每个数据集以目录分开，并在各自目录下适当建立raw/train/test/val等目录，来存放数据划分

# checkpoints

该目录用于保存模型的相关参数和日志文件

# model

存放模型的具体实现，继承`basemodel.py`中的`BaseModel`类。不同模型使用不同`module`(不同文件)来构建

## BaseMode 类

### 1、**\_\_init\_\_(self, args)**

    初始化

### 2、fit

当模型配置号之后，调用该接口进行训练，**注意**: 调用该方法之前，需调用`model.criterion = criterion`和`model.optimizer = optimizer`来设置损失函数和优化器

### 3、val_one_epoch

在训练集上执行计算一次loss和metrics相关量化指标，默认实现是在批处理数据时调用`compute_loss`和`compute_metric`然后在所有批处理的结果上平均

### 4、predict

执行预测过程，并从模型的数据结果中过滤掉最终不需要的中间结果

### 5、step

在批处理数据上执行一次前向操作，在这里进行`input`和`target`的封装处理，**该类自己实现概率很大，因为从dataloader出来的结构，有时并不是最终输入模型的数据，因此需要再次进行处理**

### 6、compute_metrics

计算metrics，注意这里面调用了`get_pred_for_metric`来从模型的输出pred中来选择出计算metric的输出，因为，在训练模型时，可能模型多个输出，比如中间结果，但最终计算metric的可能只是最后一层的结果。

### 7、compute_loss

计算loss，这里调用了`get_pred_for_loss`,原理同`get_pred_for_metric`

### 8、save_state_dict

保存模型参数。默认实现`model`、`optimizer`、`lr_scheduler`和`epoch`的`state_dict`,且保存策略是饱醉验证机上loss最优的模型和最后一次模型

### 9、restore_state_dict

从保存的state_dict初始化`model`

### 10、get_pred_for_loss

获取用于计算`loss`的pred

### 11、get_pred_for_metric

获取用于计算`metric`的pred

### 12、get_pred_for_vis

获取可视化的pred

### 13、get_metric_info

将保存metrics的字典结构格式化为"metric_name1=value, metric_name2=value..."的形式

### 14、initialize

初始化模型参数，这里采用的通用初始化结构，如果模型有一些特殊初始化，请在子类中先调用`super().initialize()`，然后实现特俗初始化。该方法可在模型的`__init__`末尾调用，或者trainner中调用

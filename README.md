## Introduction

xgboost for pyspark

python3.5 spark2.4.x xgboost0.9

## python install

pip install PyXGBoost

## code examples:

```python
from pyspark.sql import SparkSession
from PyXGBoost import PyXGBoostClassifier, PyXGBoostClassificationModel
spark = SparkSession \
    .builder \
    .appName("pyspark xgboost") \
    .getOrCreate()

df=spark.read.csv("src/main/resources/iris.csv",schema="sepal_length double, sepal_width double, petal_length double,petal_width double,label int")
df=df.fillna(0)
#same as xgboost param map
params0 = {
    "objective" :"binary:logistic"
    , "eta" : 0.01
    , "max_depth" : 6
    , "min_child_weight" : 50
    , "colsample_bytree" : 0.5
    , "silent" : 0
    , "seed" : 12345
}

xgb=PyXGBoostClassifier(params0)

xgb.set_num_round(11) \
	.set_num_workers(11)
	
feature_names=["sepal_length","sepal_width","petal_length","petal_width"]
xgbModel=xgb.train(df,feature_names, "label")
xgbModel.saveOverwrite("hdfs://xxxx")
#xgbModel.write().overwrite().save("hdfs://xxxx")
xgbModel=PyXGBoostClassificationModel.load("hdfs://xxxx")
result_df=xgbModel.transform(df,feature_names)

```
## download jar

https://github.com/303844828/PyXGBoost/releases/tag/1.0


## submit

```shell
spark-submit --master yarn-cluster --num-executors 100 \
--jars pyspark-xgboost-1.0.jar  \
--py-files pyspark-xgboost-1.0.jar \
--files test.py
```


## 简介

pyspark版本的xgboost

## 首先执行：

pip install PyXGBoost


## 代码示例：

```python
from pyspark.sql import SparkSession
from PyXGBoost import PyXGBoostClassifier, PyXGBoostClassificationModel
spark = SparkSession \
    .builder \
    .appName("pyspark xgboost") \
    .getOrCreate()

df=spark.read.csv("src/main/resources/iris.csv",schema="sepal_length double, sepal_width double, petal_length double,petal_width double,label int")
df=df.fillna(0)
#same as xgboost param map
params0 = {
    "objective" :"binary:logistic"
    , "eta" : 0.01
    , "max_depth" : 6
    , "min_child_weight" : 50
    , "colsample_bytree" : 0.5
    , "silent" : 0
    , "seed" : 12345
}

xgb=PyXGBoostClassifier(params0)

xgb.set_num_round(11) \
	.set_num_workers(11)
	
feature_names=["sepal_length","sepal_width","petal_length","petal_width"]
xgbModel=xgb.train(df,feature_names, "label")
xgbModel.saveOverwrite("hdfs://xxxx")
#xgbModel.write().overwrite().save("hdfs://xxxx")
xgbModel=PyXGBoostClassificationModel.load("hdfs://xxxx")
result_df=xgbModel.transform(df,feature_names)

```

## 下载jar包：

https://github.com/303844828/PyXGBoost/releases/tag/1.0


## 提交

命令需要在两个地方带上jar包，示例：

```shell
spark-submit --master yarn-cluster --num-executors 100 \
--jars pyspark-xgboost-1.0.jar  \
--py-files pyspark-xgboost-1.0.jar \
--files test.py      
```
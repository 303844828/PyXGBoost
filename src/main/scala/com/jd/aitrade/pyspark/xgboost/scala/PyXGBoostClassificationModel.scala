package com.jd.aitrade.pyspark.xgboost.scala

import com.jd.aitrade.pyspark.xgboost.JSONFormat
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

/**
  * Created by IntelliJ IDEA.
  *
  * @author : zhaobin
  * @date : 2019/10/17  
  **/
class PyXGBoostClassificationModel(val model:XGBoostClassificationModel) extends Serializable {

  def processRow(row:Row)={
    var rowArray=ArrayBuffer[Any]()
    var i=0
    for (i<-0 until row.length){
      var iVal=row(i)
      if (iVal.isInstanceOf[org.apache.spark.ml.linalg.Vector]){
        iVal=iVal.asInstanceOf[org.apache.spark.ml.linalg.Vector].toDense
      }
      rowArray+=iVal
    }
    Row(rowArray:_*)
  }

  def transform(predictData : org.apache.spark.sql.Dataset[_],features : java.util.List[String]):DataFrame ={
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    val vectorAssembler = new VectorAssembler().
      setInputCols(features.asScala.toArray).
      setOutputCol("tempFeatures")
    println("transform features:"+JSONFormat.format(features.asScala.toArray))
    println("transformData=====")
    predictData.show(100)
    val xgbInput = vectorAssembler.transform(predictData)//.select("tempFeatures")
    println("transformInput=====")
    xgbInput.show(100)
    val xgbInputDense=spark.createDataFrame(xgbInput.rdd.map(row=>{processRow(row)}),xgbInput.schema).withColumnRenamed("tempFeatures","features")
    println("transformInputDenseWithRenamed======")
    xgbInputDense.show(100)

    return model.transform(xgbInputDense)
  }


  def summary : String =model.summary.toString()

  def setLeafPredictionCol(value : scala.Predef.String) : PyXGBoostClassificationModel={
    model.setLeafPredictionCol(value)
    this
  }
  def setContribPredictionCol(value : scala.Predef.String) : PyXGBoostClassificationModel.this.type = {
    model.setContribPredictionCol(value)
    this
  }
  def setTreeLimit(value : scala.Int) : PyXGBoostClassificationModel.this.type = {
    model.setTreeLimit(value)
    this
  }
  def setInferBatchSize(value : scala.Int) : PyXGBoostClassificationModel.this.type = {
    model.setInferBatchSize(value)
    this
  }
  def predict(features : org.apache.spark.ml.linalg.Vector) : scala.Double = {
    model.predict(features)
  }

  def save(path:String)  = {
    model.save(path)
  }
  def saveOverwrite(path:String)={
    model.write.overwrite().save(path)
  }
  def write()={
    model.write
  }
}

object PyXGBoostClassificationModel{
  def load(path:String):PyXGBoostClassificationModel={
    val xGBoostClassificationModel:XGBoostClassificationModel =XGBoostClassificationModel.load(path)
    new PyXGBoostClassificationModel(xGBoostClassificationModel)
  }

}
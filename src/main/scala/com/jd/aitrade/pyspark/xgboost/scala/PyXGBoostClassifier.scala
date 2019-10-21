package com.jd.aitrade.pyspark.xgboost.scala

import java.util.{List => JList}
import java.util.{Map => JMap}

import com.alibaba.fastjson.JSON
import com.jd.aitrade.pyspark.xgboost.JSONFormat
import com.jd.aitrade.pyspark.xgboost.scala.XGBoostClassifier4JavaWrapper.logger
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._

case class TrainData(features:DenseVector,label:Int)
case class PredictData(features:DenseVector)

class PyXGBoostClassifier(params : JMap[String,scala.Any]){

  val xGBoostClassifier:XGBoostClassifier=if (this.params==null) new XGBoostClassifier() else new XGBoostClassifier(toScalaParams(params))
  logger.error("params=" + JSONFormat.format(params))

  private def toScalaParams(params: JMap[String,scala.Any]): Map[String,Any] ={
    logger.error("params:=="+JSONFormat.format(params))
    val paramsScala=scala.collection.mutable.Map[String,scala.Any]()
    val iter:java.util.Iterator[JMap.Entry[String,Any]] =params.entrySet().iterator();
    while(iter.hasNext){
      val entry:JMap.Entry[String,scala.Any]=iter.next()
      if ("seed".equals(entry.getKey)){
        logger.error("params:=="+JSONFormat.format(entry.getValue))
        //val longValue=Integer.valueOf(String.valueOf(entry.getValue)).toLong
        paramsScala(entry.getKey)=entry.getValue.toString.toLong
      }else{
        paramsScala(entry.getKey)=entry.getValue
      }
    }
    val immutableParams=paramsScala.toMap[String,Any]
    logger.error("scala params"+immutableParams.toString())
    return immutableParams
  }




  def train(ds: Dataset[_], features: JList[String], labelCol: String): PyXGBoostClassificationModel = {
    val spark=SparkSession.builder().getOrCreate()
    import spark.implicits._
    println("ds=" + ds)
    println("featureCol=" + JSONFormat.format(features))
    println("labelCol=" + labelCol)

    val vectorAssembler = new VectorAssembler().
      setInputCols(features.asScala.toArray).
      setOutputCol("tempFeatures")
    println("features:"+JSONFormat.format(features.asScala.toArray))

    println("trainingData=====")
    ds.show(100)

    val xgbInput = vectorAssembler.transform(ds).select("tempFeatures", labelCol)

    println("xgbInput=====")
    xgbInput.show(100)

    val xgbInputDense=xgbInput.map(row=>{TrainData(row.getAs[org.apache.spark.ml.linalg.Vector]("tempFeatures").toDense,row.getAs[Integer](labelCol))})

    val xgbInputDenseWithRenamed=xgbInputDense.withColumnRenamed("label",labelCol)

    println("xgbInputDenseWithRenamed======")
    xgbInputDenseWithRenamed.show(100,false)
    //println("xgbInputDenseWithRenamed dataSize="+xgbInputDenseWithRenamed.count())
    //println("xgbInputDenseWithRenamed dataSize="+xgbInputDenseWithRenamed.filter("label=1").count())
    xGBoostClassifier.setFeaturesCol("features")
    xGBoostClassifier.setLabelCol(labelCol)
    val xgbClassificationModel =xGBoostClassifier.fit(xgbInputDenseWithRenamed)

    println("train ok,"+xgbClassificationModel)

    new PyXGBoostClassificationModel(xgbClassificationModel)
  }

  def train(ds: Dataset[_], round: Int, nWorkers: Int, features: JList[String], labelCol: String): PyXGBoostClassificationModel = {
    xGBoostClassifier.setNumRound(round)
    xGBoostClassifier.setNumWorkers(nWorkers)
    train(ds,features,labelCol)
  }


  def setWeightCol(value : String): this.type = {
    xGBoostClassifier.setWeightCol(value)
    this
  }

  def setBaseMarginCol(value : scala.Predef.String) : this.type  = {
    xGBoostClassifier.setBaseMarginCol(value)
    this
  }
  def setNumClass(value : scala.Int) : this.type = {
    xGBoostClassifier.setNumClass(value)
    this
  }
  def setNumRound(value : scala.Int) : this.type = {
    xGBoostClassifier.setNumRound(value)
    this
  }
  def setNumWorkers(value : scala.Int) : this.type = {
    xGBoostClassifier.setNumWorkers(value)
    this
  }
  def setNthread(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setNthread(value)
    this
  }
  def setUseExternalMemory(value : scala.Boolean) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setUseExternalMemory(value)
    this
  }
  def setSilent(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSilent(value)
    this
  }
  def setMissing(value : scala.Float) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMissing(value)
    this
  }
  def setTimeoutRequestWorkers(value : scala.Long) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setTimeoutRequestWorkers(value)
    this
  }
  def setCheckpointPath(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setCheckpointPath(value)
    this
  }
  def setCheckpointInterval(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setCheckpointInterval(value)
    this
  }
  def setSeed(value : scala.Long) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSeed(value)
    this
  }
  def setEta(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setEta(value)
    this
  }
  def setGamma(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setGamma(value)
    this
  }
  def setMaxDepth(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMaxDepth(value)
    this
  }
  def setMinChildWeight(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMinChildWeight(value)
    this
  }
  def setMaxDeltaStep(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMaxDeltaStep(value)
    this
  }
  def setSubsample(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSubsample(value)
    this
  }
  def setColsampleBytree(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setColsampleBytree(value)
    this
  }
  def setColsampleBylevel(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setColsampleBylevel(value)
    this
  }
  def setLambda(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setLambda(value)
    this
  }
  def setAlpha(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setAlpha(value)
    this
  }
  def setTreeMethod(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setTreeMethod(value)
    this
  }
  def setGrowPolicy(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setGrowPolicy(value)
    this
  }
  def setMaxBins(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMaxBins(value)
    this
  }
  def setMaxLeaves(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMaxLeaves(value)
    this
  }
  def setSketchEps(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSketchEps(value)
    this
  }
  def setScalePosWeight(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setScalePosWeight(value)
    this
  }
  def setSampleType(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSampleType(value)
    this
  }
  def setNormalizeType(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setNormalizeType(value)
    this
  }
  def setRateDrop(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setRateDrop(value)
    this
  }
  def setSkipDrop(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setSkipDrop(value)
    this
  }
  def setLambdaBias(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setLambdaBias(value)
    this
  }
  def setObjective(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setObjective(value)
    this
  }
  def setObjectiveType(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setObjectiveType(value)
    this
  }
  def setBaseScore(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setBaseScore(value)
    this
  }
  def setEvalMetric(value : scala.Predef.String) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setEvalMetric(value)
    this
  }
  def setTrainTestRatio(value : scala.Double) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setTrainTestRatio(value)
    this
  }
  def setNumEarlyStoppingRounds(value : scala.Int) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setNumEarlyStoppingRounds(value)
    this
  }
  def setMaximizeEvaluationMetrics(value : scala.Boolean) : PyXGBoostClassifier.this.type = {
    xGBoostClassifier.setMaximizeEvaluationMetrics(value)
    this
  }


}


object XGBoostClassifier4JavaWrapper{

  val logger=LoggerFactory.getLogger(getClass)

}

from py4j.java_gateway import java_import
import os

from pyspark.sql import SparkSession


class PyXGBoostClassificationModel:

    def __init__(self,sqlc,jvmModel):
        self._jvmModel=jvmModel
        self._sqlc=sqlc

    def transform(self,train_df,features):
        return self._jvmModel.transform(train_df._jdf,features)

    def save(self,path):
        return self._jvmModel.save(path)

    def feature_importance(self):
        return self._jvmModel.feature_importance()
    def summary(self):
        return self._jvmModel.summary

    def set_leaf_prediction_col(self,value):
        self._jvmModel.setLeafPredictionCol(value)
        return self

    def set_contrib_prediction_col(self,value ):
        self._jvmModel.setContribPredictionCol(value)
        return self

    def set_tree_limit(self,value):
        self._jvmModel.setTreeLimit(value)
        return self

    def set_infer_batch_size(self,value):
        self._jvmModel.setInferBatchSize(value)
        return self

    def predict(self,features ):
        self._jvmModel.predict(features)
        return self

    def save(self,path):
        self._jvmModel.save(path)
        return self

    def saveOverwrite(self,path):
        self._jvmModel.saveOverwrite(path)
        return self

    def write(self):
        return self._jvmModel.write()

    @staticmethod
    def load(path):
        sqlc = SparkSession \
            .builder \
            .getOrCreate()
        java_import(sqlc._jvm, "com.jd.aitrade.pyspark.xgboost.scala.PyXGBoostClassifier")
        java_import(sqlc._jvm, "com.jd.aitrade.pyspark.xgboost.scala.PyXGBoostClassificationModel")
        return PyXGBoostClassificationModel(sqlc,sqlc._jvm.PyXGBoostClassificationModel.load(path))


class PyXGBoostClassifier:

    def __init__(self,params):
        sqlc = SparkSession \
            .builder \
            .getOrCreate()

        java_import(sqlc._jvm, "com.jd.aitrade.pyspark.xgboost.scala.PyXGBoostClassifier")
        java_import(sqlc._jvm, "com.jd.aitrade.pyspark.xgboost.scala.PyXGBoostClassificationModel")

        self._sqlc=sqlc
        self._xgb=sqlc._jvm.PyXGBoostClassifier(params)

    # def train(self,df,round,nworkers,features,label_col):
    #     return PyXGBoostClassificationModel(self._sqlc,self._xgb.train(df._jdf ,round,  nworkers, features, label_col))

    def train(self,df,features,label_col):
        return PyXGBoostClassificationModel(self._sqlc,self._xgb.train(df._jdf , features, label_col))

    def set_weight_col(self,value):
        self._xgb.setWeightCol(value)
        return self

    def set_base_margin_col(self,value):
        self._xgb.setBaseMarginCol(value)
        return self

    def set_num_class(self,value):
        self._xgb.setNumClass(value)
        return self

    def set_num_round(self,value):
        self._xgb.setNumRound(value)
        return self

    def set_num_workers(self,value):
        self._xgb.setNumWorkers(value)
        return self

    def set_nthread(self,value):
        self._xgb.setNthread(value)
        return self

    def set_use_external_memory(self,value):
        self._xgb.setUseExternalMemory(value)
        return self

    def set_silent(self,value):
        self._xgb.setSilent(value)
        return self

    def set_missing(self,value):
        self._xgb.setMissing(value)
        return self

    def set_timeout_request_workers(self,value):
        self._xgb.setTimeoutRequestWorkers(value)
        return self

    def set_checkpoint_path(self,value):
        self._xgb.setCheckpointPath(value)
        return self

    def set_checkpoint_interval(self,value):
        self._xgb.setCheckpointInterval(value)
        return self

    def set_seed(self,value):
        self._xgb.setSeed(value)
        return self

    def set_eta(self,value):
        self._xgb.setEta(value)
        return self

    def set_gamma(self,value):
        self._xgb.setGamma(value)
        return self

    def set_max_depth(self,value):
        self._xgb.setMaxDepth(value)
        return self

    def set_min_child_weight(self,value):
        self._xgb.setMinChildWeight(value)
        return self

    def set_max_delta_step(self,value):
        self._xgb.setMaxDeltaStep(value)
        return self

    def set_subsample(self,value):
        self._xgb.setSubsample(value)
        return self

    def set_colsample_bytree(self,value):
        self._xgb.setColsampleBytree(value)
        return self

    def set_colsample_bylevel(self,value):
        self._xgb.setColsampleBylevel(value)
        return self

    def set_lambda(self,value):
        self._xgb.setLambda(value)
        return self

    def set_alpha(self,value):
        self._xgb.setAlpha(value)
        return self

    def set_tree_method(self,value):
        self._xgb.setTreeMethod(value)
        return self

    def set_grow_policy(self,value):
        self._xgb.setGrowPolicy(value)
        return self

    def set_max_bins(self,value):
        self._xgb.setMaxBins(value)
        return self

    def set_max_leaves(self,value):
        self._xgb.setMaxLeaves(value)
        return self

    def set_sketch_eps(self,value):
        self._xgb.setSketchEps(value)
        return self

    def set_scale_pos_weight(self,value):
        self._xgb.setScalePosWeight(value)
        return self

    def set_sample_type(self,value):
        self._xgb.setSampleType(value)
        return self

    def set_normalize_type(self,value):
        self._xgb.setNormalizeType(value)
        return self

    def set_rate_drop(self,value):
        self._xgb.setRateDrop(value)
        return self

    def set_skip_drop(self,value):
        self._xgb.setSkipDrop(value)
        return self

    def set_lambda_bias(self,value):
        self._xgb.setLambdaBias(value)
        return self

    def set_objective(self,value):
        self._xgb.setObjective(value)
        return self

    def set_objective_type(self,value):
        self._xgb.setObjectiveType(value)
        return self

    def set_base_score(self,value):
        self._xgb.setBaseScore(value)
        return self

    def set_eval_metric(self,value):
        self._xgb.setEvalMetric(value)
        return self

    def set_train_test_ratio(self,value):
        self._xgb.setTrainTestRatio(value)
        return self

    def set_num_early_stopping_rounds(self,value):
        self._xgb.setNumEarlyStoppingRounds(value)
        return self

    def set_maximize_evaluation_metrics(self,value):
        self._xgb.setMaximizeEvaluationMetrics(value)
        return self




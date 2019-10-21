package com.jd.aitrade.pyspark.xgboost;

import com.alibaba.fastjson.JSON;

/**
 * Created by IntelliJ IDEA.
 *
 * @author : zhaobin
 * @date : 2019/10/15
 **/
public class JSONFormat {

    public static String format(Object obj){
        return JSON.toJSONString(obj);
    }
}

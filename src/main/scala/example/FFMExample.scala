package example

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.intel.imllib.ffm.classification._
import org.apache.spark.{SparkConf, SparkContext}

object FFMExample extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("FFMExample"))

    if (args.length != 10) {
      //FFMTrain /tmp/youlei/training_data 2 50 0.05 0.000001 false false 5 /tmp/youlei/ffm/model
      println("FFMTrain <training_file> <k> <iterations> <eta> <lambda> <normal> <random> <early_stopping> <threshold> <model_file>")
    }

    val data = sc.textFile(args(0)).map(_.split("\\s")).map(x => {
      val y = if (x(0).toInt > 0) 1.0 else -1.0
      val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
        (x(0).toInt, x(1).toInt, x(2).toDouble)
      })
      (y, nodeArray)
    }).repartition(4)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training, validation) = (splits(0), splits(1))

    //sometimes the max feature/field number would be different in training/validation dataSet,
    // so use the whole dataset to get the max feature/field number
    println(s"training: ${training.count}")
    println(s"test: ${validation.count}")

    val m = data.flatMap(x => x._2).map(_._1).max() //+ 1
    val n = data.flatMap(x => x._2).map(_._1).max() //+ 1

    val ffm: FFMModel = FFMWithAdaGrad.train(training, validation, m, n, k = args(1).toInt,
      iterations = args(2).toInt, eta = args(3).toDouble, lambda = args(4).toDouble,
      normalization = args(5).toBoolean, random = args(6).toBoolean, earlyStopping = args(7).toInt, threshold = args(8).toDouble, "sgd")
    //    val scores: RDD[(Double, Double)] = validation.map(x => {
    //      val p = ffm.predict(x._2, if (args(5).toBoolean) 1.0 / x._2.map { case (field, feature, value) => Math.pow(value, 2) }.sum else 1.0)
    //      val ret = if (p >= 0.5) 1.0 else -1.0
    //      (ret, x._1)
    //    })
    //    val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()
    //    println(s"accuracy = $accuracy")

    //    ffm.save(sc, args(7))
    //    val sameffm = FFMModel.load(sc, args(7))

    sc.stop()
  }
}


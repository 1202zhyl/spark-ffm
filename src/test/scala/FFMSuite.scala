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

package com.intel.imllib.ffm

import com.intel.imllib.ffm.classification._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.specs2._

object FFMSuite extends Specification {
  def is =
    s2"""
    FFMSuite $e1
  """

  val trainPath = "data/ffm/a9a_ffm"
  val modelPath = "model/ffm"
  val k = 2
  val iter = 3
  val eta = 0.01
  val lambda = 0.00002

  val target_accuracy = 0.8

  def e1 = {
    val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("FFMSuite"))

    val data = sc.textFile(trainPath).map(_.split("\\s")).map(x => {
      val y = if (x(0).toInt > 0) 1.0 else -1.0
      val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
        (x(0).toInt, x(1).toInt, x(2).toDouble)
      })
      (y, nodeArray)
    }).repartition(4)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (training: RDD[(Double, Array[(Int, Int, Double)])], testing) = (splits(0), splits(1))

    //sometimes the max feature/field number would be different in training/testing dataset,
    // so use the whole dataset to get the max feature/field number
    val m = data.flatMap(x => x._2).map(_._1).collect.max //+ 1
    val n = data.flatMap(x => x._2).map(_._2).collect.max //+ 1

    val ffm: FFMModel = FFMWithAdaGrad.train(training, testing, m, n, k = k, iterations = iter,
      eta = eta, lambda = lambda, normalization = false, false, 3, "adagrad")

    val scores1: RDD[(Double, Double)] = testing.map(x => {
      val p = ffm.predict(x._2, 1.0 / x._2.length)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy1 = scores1.filter(x => x._1 == x._2).count().toDouble / scores1.count()
    println(s"accuracy = $accuracy1")

    ffm.save(sc, modelPath)
    val sameffm = FFMModel.load(sc, modelPath)

    val scores2: RDD[(Double, Double)] = testing.map(x => {
      val p = sameffm.predict(x._2, 1.0 / x._2.length)
      val ret = if (p >= 0.5) 1.0 else -1.0
      (ret, x._1)
    })
    val accuracy2 = scores2.filter(x => x._1 == x._2).count().toDouble / scores2.count()
    println(s"accuracy = $accuracy2")

    sc.stop()

    (accuracy1 must be_>(target_accuracy)) and (accuracy2 must be_>(target_accuracy)) and (ffm.numFeatures must_== sameffm.numFeatures) and (ffm.numFields must_== sameffm.numFields) and (ffm.k must_== sameffm.k) and (ffm.n_iters must_== sameffm.n_iters) and (ffm.eta must_== sameffm.eta) and (ffm.lambda must_== sameffm.lambda) and (ffm.isNorm must_== sameffm.isNorm) and (ffm.random must_== sameffm.random) and (ffm.weights must_== sameffm.weights) and (ffm.sgd must_== sameffm.sgd)
  }
}


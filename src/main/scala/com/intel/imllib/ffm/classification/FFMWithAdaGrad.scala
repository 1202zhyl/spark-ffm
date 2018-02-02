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

package com.intel.imllib.ffm.classification

import com.intel.imllib.ffm.optimization._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
  * Created by vincent on 17-1-4.
  */
/**
  *
  * @param m             number of fields of input data
  * @param n             number of features of input data
  * @param k             A the number of factors that are used for pairwise interactions, respectively.
  * @param iterations    number of iterations
  * @param eta           step size to be used for each iteration
  * @param lambda        regularization for pairwise interactions
  * @param normalization whether normalize data
  * @param random        whether randomize data
  * @param earlyStopping rounds when validation accuracy keeps going down in a row
  * @param solver        "sgd": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMWithAdaGrad(m: Int, n: Int, k: Int, iterations: Int, eta: Double, lambda: Double,
                     normalization: Boolean, random: Boolean, earlyStopping: Int, threshold: Double, solver: String) extends Serializable {

  private val sgd = setOptimizer(solver)

  println(s"get numFields: + $m + ,nunFeatures: + $n + ,numFactors: + $k")

  private def generateInitWeights(): Vector = {
    val (bias, oneWayFeatures) = (1, n)

    val W = if (sgd) {
      val tmpSize = n * m * k + oneWayFeatures + bias
      println(s"allocating: + $tmpSize")
      new Array[Double](n * m * k + oneWayFeatures + bias)
    } else {
      val tmpSize = n * m * k * 2 + oneWayFeatures + bias
      println("allocating:" + tmpSize)
      new Array[Double](n * m * k * 2 + oneWayFeatures + bias)
    }
    val coef = 1.0 / Math.sqrt(k)
    val random = new Random()
    var position = 0
    if (sgd) {
      for (j <- 0 until n; f <- 0 until m; d <- 0 until k) {
        W(position) = coef * random.nextDouble()
        position += 1
      }
    } else {
      for (j <- 0 until n; f <- 0 until m; d <- 0 until 2 * k) {
        W(position) = if (d < k) coef * random.nextDouble() else 1.0
        position += 1
      }
    }

    /* for one way features */
    for (p <- 0 until oneWayFeatures) {
      W(position) = 0.0
      position += 1
    }

    /* for bias */
    W(position) = 0.0

    Vectors.dense(W)
  }

  /**
    * Create a FFMModle from an encoded vector.
    */
  private def createModel(weights: Vector): FFMModel = {
    val parameters = weights.toArray
    new FFMModel(n, m, k, iterations, eta, lambda, normalization, random, parameters, sgd)
  }

  /**
    * Run the algorithm with the configured parameters on an input RDD
    * of FFMNode entries.
    */
  def run(training: RDD[(Double, Array[(Int, Int, Double)])], validation: RDD[(Double, Array[(Int, Int, Double)])]): FFMModel = {
    val gradient = new FFMGradient(m, n, k, sgd)
    val optimizer = new GradientDescentFFM(gradient, null, k, iterations, eta, lambda, normalization, random, earlyStopping, threshold)

    val initWeights = generateInitWeights()
    val weights = optimizer.optimize(training, validation, initWeights, sgd)
    createModel(weights)
  }

  def setOptimizer(op: String): Boolean = {
    if ("sgd" == op) true else false
  }

}

object FFMWithAdaGrad {
  /**
    * @param training      training data RDD
    * @param validation    validate data RDD
    * @param m             number of fields of input data
    * @param n             number of features of input data
    * @param k             the number of factors that are used for pairwise interactions, respectively.
    * @param iterations    number of iterations
    * @param eta           step size to be used for each iteration
    * @param lambda        regularization for pairwise interactions
    * @param normalization whether normalize data
    * @param random        whether randomize data
    * @param solver        "sgd": parallelizedSGD, parallelizedAdaGrad would be used otherwise
    * @return FFMModel
    */
  def train(training: RDD[(Double, Array[(Int, Int, Double)])], validation: RDD[(Double, Array[(Int, Int, Double)])], m: Int, n: Int,
            k: Int, iterations: Int, eta: Double, lambda: Double, normalization: Boolean, random: Boolean, earlyStopping: Int, threshold: Double,
            solver: String = "sgd"): FFMModel = {
    new FFMWithAdaGrad(m, n, k, iterations, eta, lambda, normalization, random, earlyStopping, threshold, solver)
      .run(training, validation)
  }
}

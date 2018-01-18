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

package com.intel.imllib.ffm.optimization

import breeze.linalg.{DenseVector => BDV}
import com.intel.imllib.ffm.classification._
import com.intel.imllib.util.vectorUtils._
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by vincent on 17-1-4.
  */
class GradientDescentFFM(private var gradient: FFMGradient, private var updater: Updater,
                         k: Int, iterations: Int, eta: Double, lambda: Double,
                         normalization: Boolean, random: Boolean, earlyStopping: Int) extends Optimizer {

  val sgd = true
  private var stepSize: Double = eta
  private var numIterations: Int = iterations
  private var regParam: Double = lambda
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001

  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got $step")
    this.stepSize = step
    this
  }

  /**
    * :: Experimental ::
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got $fraction")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iterations: Int): this.type = {
    require(iterations >= 0,
      s"Number of iterations must be nonnegative but got $iterations")
    this.numIterations = iterations
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got $regParam")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    * is compared to relative tolerance which means normalizing by the norm of
    * the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    * is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got $tolerance")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: FFMGradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    Array(1).toVector.asInstanceOf[Vector]

  }

  def optimize(training: RDD[(Double, Array[(Int, Int, Double)])], validation: RDD[(Double, Array[(Int, Int, Double)])], initialWeights: Vector, solver: Boolean): Vector = {
    val (weights, _) = GradientDescentFFM.parallelAdaGrad(training, validation, gradient,
      initialWeights, this.numIterations, this.stepSize, this.regParam, normalization, random, earlyStopping, solver)
    weights
  }

}

object GradientDescentFFM {

  def parallelAdaGrad(training: RDD[(Double, Array[(Int, Int, Double)])],
                      validation: RDD[(Double, Array[(Int, Int, Double)])],
                      gradient: FFMGradient,
                      initialWeights: Vector,
                      iterations: Int,
                      eta: Double,
                      lambda: Double,
                      normalization: Boolean,
                      random: Boolean,
                      earlyStopping: Int,
                      solver: Boolean): (Vector, Array[Double]) = {
    val numIterations = iterations
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    val validationAccuracyHistory = new ArrayBuffer[Double](numIterations)
    var tmpWeights = Vectors.dense(initialWeights.toArray)
    var bestWeights: Vector = tmpWeights
    var bestIteration: Int = 0
    val slices = training.getNumPartitions

    var i = 1
    var shouldStop = false
    while (!shouldStop && i <= numIterations) {
      val bcWeights = training.context.broadcast(tmpWeights)
      val (wSum, lSum) = (if (random) training.mapPartitions(Random.shuffle(_)) else training).treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          val r = if (normalization) 1.0 / v._2.map { case (field, feature, value) => Math.pow(value, 2) }.sum else 1.0
          gradient.asInstanceOf[FFMGradient].computeFFM(v._1, v._2, fromBreeze(c._1),
            r, eta / Math.sqrt(i), lambda, i, solver = solver)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level

      tmpWeights = Vectors.dense(wSum.toArray.map(_ / slices))
      stochasticLossHistory += lSum / slices
      println("iter:" + i + ", tr_loss:" + lSum / slices)

      val scores: RDD[(Double, Double)] = validation.map(x => {
        val ffm = new FFMModel(gradient.n, gradient.m, gradient.k, iterations, eta, lambda, normalization, random, tmpWeights.toArray, gradient.sgd)
        val p = ffm.predict(x._2, if (normalization) 1.0 / x._2.map { case (field, feature, value) => Math.pow(value, 2) }.sum else 1.0)
        val ret = if (p >= 0.5) 1.0 else -1.0
        (ret, x._1)
      })
      val accuracy = scores.filter(x => x._1 == x._2).count().toDouble / scores.count()

      validationAccuracyHistory.find(_ > accuracy) match {
        case Some(acc) =>
        case None =>
          bestIteration = i
          bestWeights = tmpWeights
      }

      validationAccuracyHistory += accuracy
      shouldStop = stopOrNot(validationAccuracyHistory.toList, earlyStopping)
      println(s"accuracy = $accuracy")

      i += 1
    }
    println(s"best iteration = $bestIteration")

    (bestWeights, stochasticLossHistory.toArray)
  }

  def stopOrNot(accuracyHistory: List[Double], earlyStopping: Int): Boolean = {
    var stopping = 0
    accuracyHistory.takeRight(earlyStopping + 1).reduceLeft((l1, l2) => {
      if (l1 >= l2) stopping += 1
      l2
    })

    if (stopping >= earlyStopping) true else false
  }

}

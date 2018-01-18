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

import java.io._

import breeze.linalg.{DenseVector => BDV}
import com.intel.imllib.util.Loader._
import com.intel.imllib.util.{Loader, Saveable}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.json4s.DefaultFormats
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import scala.collection.mutable
import scala.util.Random

/**
  * Created by vincent on 16-12-19.
  */
/**
  *
  * @param numFeatures number of features
  * @param numFields   number of fields
  * @param k           A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *                    one-way interactions should be used, and the number of factors that are used for pairwise
  *                    interactions, respectively.
  * @param n_iters     number of iterations
  * @param eta         step size to be used for each iteration
  * @param lambda      regularization for pairwise interations
  * @param isNorm      whether normalize data
  * @param random      whether randomize data
  * @param weights     weights of FFMModel
  * @param sgd         "true": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMModel(val numFeatures: Int,
               val numFields: Int,
               val k: Int,
               val n_iters: Int,
               val eta: Double,
               val lambda: Double,
               val isNorm: Boolean,
               val random: Boolean,
               val weights: Array[Double],
               val sgd: Boolean = true) extends Serializable with Saveable {

  /*numFeatures*/
  private val n: Int = numFeatures

  /*numFields*/
  private val m: Int = numFields

  require(n > 0 && k > 0 && m > 0)

  def randomization(l: Int, rand: Boolean): Array[Int] = {
    val order = Array.fill(l)(0)
    for (i <- 0 until l) {
      order(i) = i
    }
    if (rand) {
      val rand = new Random()
      for (i <- l - 1 to 1) {
        val tmp = order(i - 1)
        val index = rand.nextInt(i)
        order(i - 1) = order(index)
        order(index) = tmp
      }
    }
    order
  }

  def setOptimizer(op: String): Boolean = {
    if ("sgd" == op) true else false
  }

  def predict(data: Array[(Int, Int, Double)], r: Double): Double = {

    var t = weights(weights.length - 1)

    val (align0, align1) = if (sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }

    // j: feature, f: field, v: value
    val valueSize = data.length //feature length
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2 - 1
      val f1 = data(i)._1
      val v1 = data(i)._3

      t += weights(pos + j1) * v1

      ii = i + 1
      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
          val j2 = data(ii)._2
          val f2 = data(ii)._1
          val v2 = data(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 until k) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    1 / (1 + math.exp(-t))
  }

  override protected def formatVersion: String = "1.0"

  override def save(sc: SparkContext, path: String): Unit = {
    val data = FFMModel.SaveLoadV1_0.Data(numFeatures, numFields, k, n_iters, eta, lambda, isNorm, random, weights, sgd)
    FFMModel.SaveLoadV1_0.save(sc, path, data)
  }
}

object FFMModel extends Loader[FFMModel] {

  private object SaveLoadV1_0 {

    def thisFormatVersion = "1.0"

    def thisClassName = "com.intel.imllib.ffm.classification.FFMModel$SaveLoadV1_0$"

    /** Model data for model import/export */
    case class Data(numFeatures: Int, numFields: Int, k: Int,
                    n_iters: Int, eta: Double, lambda: Double, isNorm: Boolean,
                    random: Boolean, weights: Array[Double], sgd: Boolean)

    def save(sc: SparkContext, path: String, data: Data): Unit = {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      // Create JSON metadata.
      val metadata = compact(render(
        ("class" -> this.getClass.getName) ~ ("version" -> thisFormatVersion) ~
          ("numFeatures" -> data.numFeatures) ~ ("numFields" -> data.numFields)
          ~ ("n_iters" -> data.n_iters) ~ ("eta" -> data.eta) ~ ("lambda" -> data.lambda)
          ~ ("isNorm" -> data.isNorm) ~ ("random" -> data.random) ~ ("sgd" -> data.sgd)))
      sc.parallelize(Seq(metadata), 1).saveAsTextFile(metadataPath(path))

      // Create Parquet data.
      val dataRDD: DataFrame = sc.parallelize(Seq(data), 1).toDF()
      dataRDD.write.parquet(dataPath(path))
    }

    def load(sc: SparkContext, path: String): FFMModel = {
      val sqlContext = new SQLContext(sc)
      // Load Parquet data.
      val dataRDD = sqlContext.parquetFile(dataPath(path))
      // Check schema explicitly since erasure makes it hard to use match-case for checking.
      checkSchema[Data](dataRDD.schema)
      val dataArray = dataRDD.select("numFeatures", "numFields", "dim0", "dim1", "dim2", "n_iters", "eta", "lambda", "isNorm", "random", "weights", "sgd").take(1)
      assert(dataArray.length == 1, s"Unable to load FMModel data from: ${dataPath(path)}")
      val data = dataArray(0)
      val numFeatures = data.getInt(0)
      val numFields = data.getInt(1)
      val k = data.getInt(2)
      val n_iters = data.getInt(3)
      val eta = data.getDouble(4)
      val lambda = data.getDouble(5)
      val isNorm = data.getBoolean(6)
      val random = data.getBoolean(7)
      val weights = data.getAs[mutable.WrappedArray[Double]](8).toArray
      val sgd = data.getBoolean(9)
      new FFMModel(numFeatures, numFields, k, n_iters, eta, lambda, isNorm, random, weights, sgd)
    }
  }

  override def load(sc: SparkContext, path: String): FFMModel = {
    implicit val formats = DefaultFormats

    val (loadedClassName, version, metadata) = loadMetadata(sc, path)
    val classNameV1_0 = SaveLoadV1_0.thisClassName

    (loadedClassName, version) match {
      case (className, "1.0") if className == classNameV1_0 =>
        val numFeatures = (metadata \ "numFeatures").extract[Int]
        val numFields = (metadata \ "numFields").extract[Int]
        val model = SaveLoadV1_0.load(sc, path)
        assert(model.numFeatures == numFeatures,
          s"FFMModel.load expected $numFeatures features," +
            s" but model had ${model.numFeatures} featues")
        assert(model.numFields == numFields,
          s"FFMModel.load expected $numFields fields," +
            s" but model had ${model.numFields} fields")
        model

      case _ => throw new Exception(
        s"FFMModel.load did not recognize model with (className, format version):" +
          s"($loadedClassName, $version).  Supported:\n" +
          s"  ($classNameV1_0, 1.0)")
    }
  }
}


class FFMGradient(val m: Int, val n: Int, val k: Int, val sgd: Boolean = true) extends Gradient {

  private def predict(data: Array[(Int, Int, Double)], weights: Array[Double], r: Double = 1.0): Double = {

    /* weights: interactive weights + one-way weights + bias */
    var t = weights(weights.length - 1)

    val (align0, align1) = if (sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data.length //feature length
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data(i)._2 - 1
      val f1 = data(i)._1
      val v1 = data(i)._3
      ii = i + 1

      t += weights(pos + j1) * v1

      if (j1 < n && f1 < m) {
        while (ii < valueSize) {
          val j2 = data(ii)._2
          val f2 = data(ii)._1
          val v2 = data(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 until k) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    t
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFFM()")
  }

  def computeFFM(label: Double, data2: Array[(Int, Int, Double)], weights: Vector,
                 r: Double = 1.0, eta: Double, lambda: Double,
                 iter: Int, solver: Boolean = true): (BDV[Double], Double) = {
    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    val t = predict(data2, weightsArray, r)
    val expnyt = math.exp(-label * t)
    val tr_loss = math.log(1 + expnyt)
    val kappa = -label * expnyt / (1 + expnyt)
    val (align0, align1) = if (sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }

    //feature length
    val valueSize = data2.length
    var i = 0
    var ii = 0

    val r0, r1 = 0.0
    val pos = if (sgd) n * m * k else n * m * k * 2
    weightsArray(weightsArray.length - 1) -= eta * (kappa + r0 * weightsArray(weightsArray.length - 1))

    // j: feature, f: field, v: value
    while (i < valueSize) {
      val j1 = data2(i)._2 - 1
      val f1 = data2(i)._1
      val v1 = data2(i)._3

      weightsArray(pos + j1) -= eta * (v1 * kappa + r1 * weightsArray(pos + j1))
      if (j1 < n && f1 < m) {
        ii = i + 1
        while (ii < valueSize) {
          val j2 = data2(ii)._2
          val f2 = data2(ii)._1
          val v2 = data2(ii)._3
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            val wg1_index: Int = w1_index + k
            val wg2_index: Int = w2_index + k
            val kappav: Double = kappa * v
            for (d <- 0 until k) {
              val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
              val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
              if (sgd) {
                weightsArray(w1_index + d) -= eta * g1
                weightsArray(w2_index + d) -= eta * g2
              } else {
                val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
                val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
                weightsArray(w1_index + d) -= eta / math.sqrt(wg1) * g1
                weightsArray(w2_index + d) -= eta / math.sqrt(wg2) * g2
                weightsArray(wg1_index + d) = wg1
                weightsArray(wg2_index + d) = wg2

              }
            }
          }
          ii += 1
        }
      }
      i += 1
    }
    (BDV(weightsArray), tr_loss)
  }
}

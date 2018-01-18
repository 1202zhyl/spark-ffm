package com.intel.imllib.util

import breeze.linalg.{DenseVector=>BDV, SparseVector=>BSV, Vector=>BV}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

/**
  * Created by wxie6 on 25/03/2017.
  */
object vectorUtils {
  def toBreeze(mllibVec: Vector): BV[Double] = new BDV[Double](mllibVec.toDense.values)
  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray) // Can't use underlying array directly, so make a new one
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}

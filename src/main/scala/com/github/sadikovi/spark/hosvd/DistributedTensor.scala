/*
 * Copyright 2016 sadikovi
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.sadikovi.spark.hosvd

import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * [[DistributedTensor]] is an RDD-based tensor, can be in compressed format similar to
 * CoordinateMatrix class in Spark. Dimensions are lazily computed, if none provided.
 */
class DistributedTensor(
    @transient val entries: RDD[TensorEntry],
    private var rows: Int,
    private var cols: Int,
    private var layers: Int)
  extends Tensor {

  // Since tensor can be very large, we have to ensure that underlying RDD is persisted
  persistTensor(StorageLevel.MEMORY_AND_DISK)

  // Whether or not dimensions are already computed
  private var computed: Boolean = false

  /** Alternative constructor leaving tensor dimensions to be determined automatically. */
  def this(entries: RDD[TensorEntry]) = this(entries, 0, 0, 0)

  /** Computes size dynamicaly. */
  private def computeSize() {
    // once method is called we know, that dimensions are recomputed
    computed = true

    val (imax, jmax, kmax) = entries.
      aggregate((0, 0, 0))(
        (U, V) => (math.max(U._1, V.i), math.max(U._2, V.j), math.max(U._3, V.k)),
        (U1, U2) => (math.max(U1._1, U2._1), math.max(U1._2, U2._2), math.max(U1._3, U2._3))
      )
    // reassign dimensions
    rows = 1 + imax
    cols = 1 + jmax
    layers = 1 + kmax
  }

  override def numRows: Int = {
    if (rows <= 0) {
      computeSize()
    }
    rows
  }

  override def numCols: Int = {
    if (cols <= 0) {
      computeSize()
    }
    cols
  }

  override def numLayers: Int = {
    if (layers <= 0) {
      computeSize()
    }
    layers
  }

  override def unfold(direction: UnfoldDirection.Value): DistributedUnfoldResult = {
    val matrix = direction match {
      case UnfoldDirection.A1 =>
        new CoordinateMatrix(entries.map { entry =>
          MatrixEntry(entry.i, entry.j + numCols * entry.k, entry.value)
        }, numRows, numLayers * numCols)
      case UnfoldDirection.A2 =>
        new CoordinateMatrix(entries.map { entry =>
          MatrixEntry(entry.j, entry.k + numLayers * entry.i, entry.value)
        }, numCols, numRows * numLayers)
      case UnfoldDirection.A3 =>
        new CoordinateMatrix(entries.map { entry =>
          MatrixEntry(entry.k, entry.i + numRows * entry.j, entry.value)
        }, numLayers, numCols * numRows)
      case otherMode =>
        throw new IllegalArgumentException(s"Unrecognized unfolding mode $otherMode")
    }
    DistributedUnfoldResult(matrix, direction)
  }

  /** Persist tensor with provided level, if none set */
  private def persistTensor(level: StorageLevel): Unit = {
    if (entries.getStorageLevel == StorageLevel.NONE) {
      entries.persist(level)
    }
  }

  override def hosvd(k1: Int, k2: Int, k3: Int): Tensor = {
    val unfoldingA1 = unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult].matrix.
      toIndexedRowMatrix
    val unfoldingA2 = unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult].matrix.
      toIndexedRowMatrix
    val unfoldingA3 = unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].matrix.
      toIndexedRowMatrix

    val svd1 = unfoldingA1.computeSVD(k1, computeU = true)
    val svd2 = unfoldingA2.computeSVD(k2, computeU = true)
    val svd3 = unfoldingA3.computeSVD(k3, computeU = true)

    val U1 = svd1.U.toBlockMatrix.transpose
    val mult1 = U1.multiply(unfoldingA1.toBlockMatrix)
    val tensor1 = DistributedTensor.fold(mult1.toCoordinateMatrix, UnfoldDirection.A1,
      mult1.numRows.toInt, numCols, numLayers)

    val U2 = svd2.U.toBlockMatrix.transpose
    val mult2 = U2.multiply(tensor1.unfold(UnfoldDirection.A2).
      asInstanceOf[DistributedUnfoldResult].matrix.toBlockMatrix)
    val tensor2 = DistributedTensor.fold(mult2.toCoordinateMatrix, UnfoldDirection.A2,
      tensor1.numRows, mult2.numRows.toInt, tensor1.numLayers)

    val U3 = svd3.U.toBlockMatrix.transpose
    val mult3 = U3.multiply(tensor2.unfold(UnfoldDirection.A3).
      asInstanceOf[DistributedUnfoldResult].matrix.toBlockMatrix)
    val tensor3 = DistributedTensor.fold(mult3.toCoordinateMatrix, UnfoldDirection.A3,
      tensor2.numRows, tensor2.numCols, mult3.numRows.toInt)

    tensor3
  }
}

object DistributedTensor extends TensorLike {
  private def failDimensionsCheck(
      matrix: CoordinateMatrix,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Unit = {
    throw new IllegalArgumentException("Failed to match dimensions from coordinate matrix to " +
      s"tensor using direction $direction. Cannot convert ${matrix.numRows}x${matrix.numCols} " +
      s"into ${rows}x${cols}x${layers}")
  }

  override def fold(
      matrix: CoordinateMatrix,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Tensor = {
    val rdd = direction match {
      case UnfoldDirection.A1 =>
        if (!(matrix.numRows == rows && matrix.numCols == cols.toLong * layers)) {
          failDimensionsCheck(matrix, direction, rows, cols, layers)
        }
        matrix.entries.map { entry =>
          val i = entry.i.toInt
          val j = entry.j.toInt % cols
          val k = (entry.j - j).toInt / cols
          TensorEntry(i, j, k, entry.value)
        }
      case UnfoldDirection.A2 =>
        if (!(matrix.numRows == cols && matrix.numCols == rows.toLong * layers)) {
          failDimensionsCheck(matrix, direction, rows, cols, layers)
        }
        matrix.entries.map { entry =>
          val j = entry.i.toInt
          val k = entry.j.toInt % layers
          val i = (entry.j - k).toInt / layers
          TensorEntry(i, j, k, entry.value)
        }
      case UnfoldDirection.A3 =>
        if (!(matrix.numRows == layers && matrix.numCols == cols.toLong * rows)) {
          failDimensionsCheck(matrix, direction, rows, cols, layers)
        }
        matrix.entries.map { entry =>
          val i = entry.j.toInt % rows
          val j = (entry.j - i).toInt / rows
          val k = entry.i.toInt
          TensorEntry(i, j, k, entry.value)
        }
      case otherMode =>
        throw new IllegalArgumentException(s"Unrecognized unfolding mode $otherMode")
    }
    new DistributedTensor(rdd, rows, cols, layers)
  }
}

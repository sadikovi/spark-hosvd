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

package com.github.sadikovi.hosvd

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
  if (entries.getStorageLevel == StorageLevel.NONE) {
    entries.persist(StorageLevel.MEMORY_AND_DISK)
  }

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
        throw new Exception(s"Unrecognized unfolding mode $otherMode")
    }
    DistributedUnfoldResult(matrix)
  }
}

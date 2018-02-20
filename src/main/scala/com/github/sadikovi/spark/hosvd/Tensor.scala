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

import org.apache.spark.mllib.linalg.{Matrix, SingularValueDecomposition}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRowMatrix}

/** Tensor entry for 'i' row, 'j' column and 'k' layer with value 'value' */
case class TensorEntry(i: Int, j: Int, k: Int, value: Double) {
  override def hashCode: Int = {
    var result = value.hashCode
    result = 31 * result + i
    result = 31 * result + j
    result = 31 * result + k
    result
  }

  override def equals(that: Any): Boolean = {
    if (that == null || getClass() != that.getClass()) return false
    val entry = that.asInstanceOf[TensorEntry]
    entry.i == i && entry.j == j && entry.k == k && entry.value == value
  }

  override def toString: String = {
    s"[($i, $j, $k) -> $value]"
  }
}

/**
 * [[Tensor]] represents 3rd order rows x columns x layers tensor.
 */
abstract class Tensor extends Serializable {
  /** Number of rows in tensor */
  def numRows: Int

  /** Number of columns in tensor */
  def numCols: Int

  /** Number of layers in tensor */
  def numLayers: Int

  /** Get layer slice of tensor */
  def getLayer(layer: Int): Matrix

  /**
   * Unfold current tensor with direction.
   * @param direction direction to unfold
   * @return unfolding result as instance of `UnfoldResult`
   */
  def unfold(direction: UnfoldDirection.Value): UnfoldResult

  /**
   * High order SVD for current tensor, returns core tensor for dimensions specified.
   * @param k1 number of singular values to keep for unfolding A1
   * @param k2 number of singular values to keep for unfolding A2
   * @param k3 number of singular values to keep for unfolding A3
   * @return core tensor
   */
  def hosvd(k1: Int, k2: Int, k3: Int): Tensor

  /**
   * Unfold by specified direction and compute svd on the unfolding.
   * @param k number of singular values to keep
   * @param direction unfolding direction
   * @return singular value decomposition
   */
  def computeSVD(
      k: Int,
      direction: UnfoldDirection.Value): SingularValueDecomposition[IndexedRowMatrix, Matrix]

  override def toString(): String = {
    s"${getClass.getSimpleName}[$numRows x $numCols x $numLayers]"
  }
}

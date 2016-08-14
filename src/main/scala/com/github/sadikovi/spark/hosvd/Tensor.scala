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

import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix

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

  override def toString(): String = {
    s"${getClass.getSimpleName}[$numRows x $numCols x $numLayers]"
  }
}

/**
 * Trait [[TensorLike]] for companion objects to create new tensor.
 */
trait TensorLike {
  /**
   * Fold matrix by provided direction into tensor.
   * Dimensions are specified must be checked against actual matrix size and direction.
   * @param matrix matrix to fold
   * @param direction direction to fold (usually direction of unfolding for matrix)
   * @param rows number of rows for created tensor
   * @param cols number of columns for created tensor
   * @param layers number of layers for created tensor
   */
  def fold(
      matrix: CoordinateMatrix,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Tensor
}

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

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql.{Dataset, DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel

/**
 * [[DistributedTensor]] is an Dataset-based tensor, can be in compressed format similar to
 * CoordinateMatrix class in Spark. Dimensions are lazily computed, if none provided.
 */
class DistributedTensor(
    entries: Dataset[TensorEntry],
    private var rows: Int,
    private var cols: Int,
    private var layers: Int)
  extends Tensor {

  @transient val data: DataFrame = entries.toDF
  @transient val spark = data.sparkSession
  import spark.implicits._

  /** Alternative constructor leaving tensor dimensions to be determined automatically. */
  def this(entries: Dataset[TensorEntry]) = this(entries, -1, -1, -1)

  /** Return underlying tensor data */
  def tensorEntries: Dataset[TensorEntry] = data.as[TensorEntry]

  /**
   * Computes size dynamicaly.
   * Once method is called we know, that dimensions are recomputed.
   */
  private def computeSize() {
    val (imax, jmax, kmax) =
      data.select(max("i"), max("j"), max("k")).
      as[(Int, Int, Int)].
      first
    // reassign dimensions
    rows = 1 + imax
    cols = 1 + jmax
    layers = 1 + kmax
  }

  override def numRows: Int = {
    if (rows < 0) {
      computeSize()
    }
    rows
  }

  override def numCols: Int = {
    if (cols < 0) {
      computeSize()
    }
    cols
  }

  override def numLayers: Int = {
    if (layers < 0) {
      computeSize()
    }
    layers
  }

  override def getLayer(layer: Int): Matrix = {
    require(layer >= 0 && layer < numLayers,
      s"Invalid layer $layer of tensor (total layers $numLayers)")
    val rdd = data.
      filter(col("k") === layer).
      select(col("i"), col("j"), col("value")).
      as[MatrixEntry].rdd
    new CoordinateMatrix(rdd, numRows, numCols).toBlockMatrix.toLocalMatrix
  }

  override def unfold(direction: UnfoldDirection.Value): UnfoldResult = {
    val matrix = direction match {
      case UnfoldDirection.A1 =>
        val rdd = data.
          select(col("i").as("i"), (col("j") + lit(numCols) * col("k")).as("j"), col("value")).
          as[MatrixEntry].
          rdd
        new CoordinateMatrix(rdd, numRows, numLayers * numCols)
      case UnfoldDirection.A2 =>
        val rdd = data.
          select(col("j").as("i"), (col("k") + lit(numLayers) * col("i")).as("j"), col("value")).
          as[MatrixEntry].
          rdd
        new CoordinateMatrix(rdd, numCols, numRows * numLayers)
      case UnfoldDirection.A3 =>
        val rdd = data.
          select(col("k").as("i"), (col("i") + lit(numRows) * col("j")).as("j"), col("value")).
          as[MatrixEntry].
          rdd
        new CoordinateMatrix(rdd, numLayers, numCols * numRows)
      case otherMode =>
        throw new IllegalArgumentException(s"Unrecognized unfolding mode $otherMode")
    }
    DistributedUnfoldResult(matrix, direction)
  }

  /** Persist tensor with provided level, if none set */
  def persist(level: StorageLevel = StorageLevel.MEMORY_AND_DISK): Unit = {
    if (data.storageLevel == StorageLevel.NONE) {
      data.persist(level)
    }
  }

  /** Unpersist tensor entries */
  def unpersist(): Unit = {
    data.unpersist()
  }

  override def hosvd(k1: Int, k2: Int, k3: Int): Tensor = {
    persist()

    val unfoldingA1 = unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult].matrix
    val unfoldingA2 = unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult].matrix
    val unfoldingA3 = unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].matrix

    val svd1 = DistributedTensor.computeSVD(unfoldingA1, k1,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)
    val svd2 = DistributedTensor.computeSVD(unfoldingA2, k2,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)
    val svd3 = DistributedTensor.computeSVD(unfoldingA3, k3,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)

    val U1 = svd1.U.toBlockMatrix.transpose
    val mult1 = U1.multiply(unfoldingA1.toBlockMatrix)
    val tensor1 = DistributedTensor.fold(spark, mult1.toCoordinateMatrix, UnfoldDirection.A1,
      mult1.numRows.toInt, numCols, numLayers)

    val U2 = svd2.U.toBlockMatrix.transpose
    val mult2 = U2.multiply(tensor1.unfold(UnfoldDirection.A2).
      asInstanceOf[DistributedUnfoldResult].matrix.toBlockMatrix)
    val tensor2 = DistributedTensor.fold(spark, mult2.toCoordinateMatrix, UnfoldDirection.A2,
      tensor1.numRows, mult2.numRows.toInt, tensor1.numLayers)

    val U3 = svd3.U.toBlockMatrix.transpose
    val mult3 = U3.multiply(tensor2.unfold(UnfoldDirection.A3).
      asInstanceOf[DistributedUnfoldResult].matrix.toBlockMatrix)
    val tensor3 = DistributedTensor.fold(spark, mult3.toCoordinateMatrix, UnfoldDirection.A3,
      tensor2.numRows, tensor2.numCols, mult3.numRows.toInt)

    unpersist()

    tensor3
  }

  override def computeSVD(
      k: Int,
      direction: UnfoldDirection.Value): SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    val matrix = unfold(direction).asInstanceOf[DistributedUnfoldResult].matrix
    DistributedTensor.computeSVD(matrix, k,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = true)
  }
}

object DistributedTensor {
  /** Create tensor from random data */
  def rand(
      spark: SparkSession,
      rows: Int,
      cols: Int,
      layers: Int,
      numPartitions: Int = 200): DistributedTensor = {
    import spark.implicits._

    val rdd = spark.sparkContext.parallelize(0 until rows, numPartitions).flatMap { row =>
      for (col <- 0 until cols) yield (row, col)
    }.flatMap { case (row, col) =>
      for (layer <- 0 until layers) yield {
        TensorEntry(row, col, layer, new java.util.Random().nextDouble())
      }
    }
    new DistributedTensor(rdd.toDS(), rows, cols, layers)
  }

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

  def fold(
      spark: SparkSession,
      matrix: CoordinateMatrix,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Tensor = {

    import spark.implicits._

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
    new DistributedTensor(rdd.toDS, rows, cols, layers)
  }

  private[hosvd] def computeSVD(
      matrix: CoordinateMatrix,
      k: Int,
      level: StorageLevel,
      rCond: Double = 1e-9,
      computeU: Boolean = false,
      computeV: Boolean = false):
    SingularValueDecomposition[IndexedRowMatrix, Matrix] = {

    val sc = matrix.entries.sparkContext

    // whether or not input matrix should be transposed for SVD
    val transposed = matrix.numCols() > matrix.numRows()
    val extractU = computeU && !transposed || computeV && transposed

    val irm = (if (transposed) matrix.transpose else matrix).toIndexedRowMatrix
    irm.rows.persist(level)
    val svd = irm.computeSVD(k, computeU = extractU, rCond = rCond)

    // collect values for matrix U for the following conditions
    val (uarr, urows, ucols) = if (extractU) {
      val arr = svd.U.rows.collect()
      (arr, arr.length, arr.head.vector.size)
    } else {
      (null, -1, -1)
    }

    val umat = if (computeU) {
      if (transposed) {
        val V = svd.V.transpose
        val vrows = V.numRows
        val vcols = V.numCols

        val rows = V.toArray.sliding(vrows, vrows).
          zipWithIndex.
          map { case (arr, row) => IndexedRow(row, Vectors.dense(arr)) }.toSeq
        new IndexedRowMatrix(sc.parallelize(rows), vcols, vrows)
      } else {
        new IndexedRowMatrix(sc.parallelize(uarr), urows, ucols)
      }
    } else {
      null
    }

    val vmat = if (computeV) {
      if (transposed) {
        new DenseMatrix(urows, ucols, uarr.flatMap { _.vector.toArray }, true)
      } else {
        svd.V
      }
    } else {
      null
    }

    SingularValueDecomposition(umat, svd.s, vmat)
  }
}

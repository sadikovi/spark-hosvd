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

/** Unfold result for [[DistributedTensor]] */
case class DistributedUnfoldResult(
    val block: CoordinateBlock,
    private val unfoldDirection: UnfoldDirection.Value)
  extends UnfoldResult {

  override def isLocal: Boolean = false

  override def direction: UnfoldDirection.Value = unfoldDirection
}

case class DistributedHOSVD(
    private val tensor: Tensor,
    private val u: Seq[Matrix],
    private val s: Seq[Vector])
  extends HOSVD {

  require(u.length == 3, s"Expected 3 matrices, found ${u.length}")
  require(s.length == 3, s"Expected 3 vectors, found ${s.length}")

  override def coreTensor: Tensor = tensor

  override def leftSingularVectors(direction: UnfoldDirection.Value): Matrix = {
    direction match {
      case UnfoldDirection.A1 => u(0)
      case UnfoldDirection.A2 => u(1)
      case UnfoldDirection.A3 => u(2)
    }
  }

  override def singularValues(direction: UnfoldDirection.Value): Vector = {
    direction match {
      case UnfoldDirection.A1 => s(0)
      case UnfoldDirection.A2 => s(1)
      case UnfoldDirection.A3 => s(2)
    }
  }
}

/**
 * [[DistributedTensor]] is an Dataset-based tensor.
 * Data is stored in compressed format similar to CoordinateMatrix class in Spark.
 * Dimensions are lazily computed, if none provided.
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
    // Reassign dimensions
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
    val ds = data.
      filter(col("k") === layer).
      select(col("i"), col("j"), col("value")).
      as[MatrixEntry]
    CoordinateBlock(ds, numRows, numCols).toLocalMatrix
  }

  override def unfold(direction: UnfoldDirection.Value): UnfoldResult = {
    // Each unfolding is described in terms of swapping coordinates for each entry,
    // below are the schemas of updating indices for each direction:
    // ===
    // A1 -> matrix of dimensions "rows x (layers * cols)":
    //  i: i
    //  j: j + cols * k
    // ===
    // A2 -> matrix of dimensions "cols x (rows * layers)":
    //  i: j
    //  j: k + layers * i
    // ===
    // A3 -> matrix of dimensions "layers x (cols * rows)":
    //  i: k
    //  j: i + rows * j

    val block = direction match {
      case UnfoldDirection.A1 =>
        val ds = data.
          select(col("i").as("i"), (col("j") + lit(numCols) * col("k")).as("j"), col("value")).
          as[MatrixEntry]
        CoordinateBlock(ds, numRows, numLayers * numCols)
      case UnfoldDirection.A2 =>
        val ds = data.
          select(col("j").as("i"), (col("k") + lit(numLayers) * col("i")).as("j"), col("value")).
          as[MatrixEntry]
        CoordinateBlock(ds, numCols, numRows * numLayers)
      case UnfoldDirection.A3 =>
        val ds = data.
          select(col("k").as("i"), (col("i") + lit(numRows) * col("j")).as("j"), col("value")).
          as[MatrixEntry]
        CoordinateBlock(ds, numLayers, numCols * numRows)
    }
    DistributedUnfoldResult(block, direction)
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

  override def hosvd(k1: Int, k2: Int, k3: Int): HOSVD = {
    persist()

    val unfoldingA1 = unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult].block
    val unfoldingA2 = unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult].block
    val unfoldingA3 = unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].block

    val svd1 = DistributedTensor.computeSVD(unfoldingA1, k1,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)
    val svd2 = DistributedTensor.computeSVD(unfoldingA2, k2,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)
    val svd3 = DistributedTensor.computeSVD(unfoldingA3, k3,
      level = StorageLevel.MEMORY_AND_DISK, computeU = true, computeV = false)

    // We optimize matrix multiplication for HOSVD.
    // Normally you would see multiplication for each folding that looks like this:
    // U.T x Ax, where
    // - U.T is transposed left singular vectors from SVD result
    // - Ax is a tensor unfolding x
    // It is faster to perform the following:
    // (Ax.T x U).T, where
    // - Ax.T is transposed unfolding
    // - U is left singular vectors

    val U1 = svd1.U
    val mat1 = unfoldingA1
    val mult1 = DistributedTensor.multiply(mat1.transpose, U1).transpose
    val tensor1 = DistributedTensor.fold(mult1, UnfoldDirection.A1,
      mult1.numRows.toInt, numCols, numLayers)

    val U2 = svd2.U
    val mat2 = tensor1.unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult].block
    val mult2 = DistributedTensor.multiply(mat2.transpose, U2).transpose
    val tensor2 = DistributedTensor.fold(mult2, UnfoldDirection.A2,
      tensor1.numRows, mult2.numRows.toInt, tensor1.numLayers)

    val U3 = svd3.U
    val mat3 = tensor2.unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].block
    val mult3 = DistributedTensor.multiply(mat3.transpose, U3).transpose
    val tensor3 = DistributedTensor.fold(mult3, UnfoldDirection.A3,
      tensor2.numRows, tensor2.numCols, mult3.numRows.toInt)

    unpersist()

    new DistributedHOSVD(tensor3, Seq(svd1.U, svd2.U, svd3.U), Seq(svd1.s, svd2.s, svd3.s))
  }

  override def computeSVD(
      k: Int,
      direction: UnfoldDirection.Value): SingularValueDecomposition[Matrix, Matrix] = {
    val block = unfold(direction).asInstanceOf[DistributedUnfoldResult].block
    DistributedTensor.computeSVD(block, k,
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
      block: CoordinateBlock,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Unit = {
    throw new IllegalArgumentException(
      "Failed to match dimensions from coordinate block (matrix) to tensor using direction " +
      s"$direction: ${block.numRows}x${block.numCols} => ${rows}x${cols}x${layers}")
  }

  def fold(
      block: CoordinateBlock,
      direction: UnfoldDirection.Value,
      rows: Int,
      cols: Int,
      layers: Int): Tensor = {
    import block.data.sparkSession.implicits._

    // To fold a matrix into a tensor, we need to perform the following coordinate remapping.
    // Each matrix element has (I, J) coordinates, each tensor element will have (i, j, k).
    // ===
    // A1 -> matrix.rows == rows and matrix.cols == cols * layers
    //  i: I
    //  j: J % cols
    //  k: (J - j) / cols
    // ===
    // A2 -> matrix.rows == cols and matrix.cols == rows * layers
    //  j: I
    //  k: J % layers
    //  i: (J - k) / cols
    // ===
    // A3 -> matrix.rows == layers and matrix.cols == cols * rows
    //  i: J % rows
    //  j: (J - i) / rows
    //  k: I

    val ds = direction match {
      case UnfoldDirection.A1 =>
        if (!(block.numRows == rows && block.numCols == cols.toLong * layers)) {
          failDimensionsCheck(block, direction, rows, cols, layers)
        }
        block.data.select(
          col("i").cast("int").as("i"),
          (col("j") % cols).cast("int").as("j"),
          ((col("j") - (col("j") % cols)) / cols).cast("int").as("k"),
          col("value")
        ).as[TensorEntry]
      case UnfoldDirection.A2 =>
        if (!(block.numRows == cols && block.numCols == rows.toLong * layers)) {
          failDimensionsCheck(block, direction, rows, cols, layers)
        }
        block.data.select(
          ((col("j") - (col("j") % layers)) / layers).cast("int").as("i"),
          col("i").cast("int").as("j"),
          (col("j") % layers).cast("int").as("k"),
          col("value")
        ).as[TensorEntry]
      case UnfoldDirection.A3 =>
        if (!(block.numRows == layers && block.numCols == cols.toLong * rows)) {
          failDimensionsCheck(block, direction, rows, cols, layers)
        }
        block.data.select(
          (col("j") % rows).cast("int").as("i"),
          ((col("j") - (col("j") % rows)) / rows).cast("int").as("j"),
          col("i").cast("int").as("k"),
          col("value")
        ).as[TensorEntry]
    }
    new DistributedTensor(ds, rows, cols, layers)
  }

  /**
   * Optimized method to compute SVD.
   * Based on selection of U and V, we either partially compute singular vectors or omit them.
   * Also matrix is transposed internally if number of columns is greater than number of rows
   * to speed up computation.
   */
  private[hosvd] def computeSVD(
      block: CoordinateBlock,
      k: Int,
      level: StorageLevel,
      rCond: Double = 1e-9,
      computeU: Boolean = false,
      computeV: Boolean = false):
    SingularValueDecomposition[Matrix, Matrix] = {

    // Whether or not input matrix should be transposed for SVD
    val transposed = block.numCols > block.numRows
    val extractU = computeU && !transposed || computeV && transposed

    val irm = (if (transposed) block.transpose else block).toIndexedRowMatrix
    irm.rows.persist(level)
    val svd = irm.computeSVD(k, computeU = extractU, rCond = rCond)

    // Collect values for matrix U for the following conditions
    val (uarr, urows, ucols) = if (extractU) {
      val arr = svd.U.rows.collect()
      (arr, arr.length, arr.head.vector.size)
    } else {
      (null, -1, -1)
    }

    val umat = if (computeU) {
      if (transposed) {
        svd.V
      } else {
        new DenseMatrix(urows, ucols, uarr.flatMap { _.vector.toArray }, true)
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

  private[hosvd] def multiply(block: CoordinateBlock, local: Matrix): CoordinateBlock = {
    import block.data.sparkSession.implicits._

    val rows = block.numRows
    val cols = local.numCols
    val irm = block.toIndexedRowMatrix.multiply(local)
    // copied from IndexedRowMatrix.toCoordinateMatrix to ensure that dimensions are not recomputed
    val entries = irm.rows.flatMap { row =>
      val rowIndex = row.index
      row.vector match {
        case SparseVector(size, indices, values) =>
          Iterator.tabulate(indices.length)(i => MatrixEntry(rowIndex, indices(i), values(i)))
        case DenseVector(values) =>
          Iterator.tabulate(values.length)(i => MatrixEntry(rowIndex, i, values(i)))
      }
    }
    CoordinateBlock(entries.toDS, rows, cols)
  }
}

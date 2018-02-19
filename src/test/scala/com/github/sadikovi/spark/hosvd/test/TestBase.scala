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

package com.github.sadikovi.spark.hosvd.test

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.DataFrame

import org.scalatest._

import com.github.sadikovi.spark.hosvd.DistributedTensor

trait TestBase extends Matchers {
  /** Compare two DataFrame objects */
  protected def checkDataFrame(df: DataFrame, expected: DataFrame): Boolean = {
    val got = df.collect().map(_.toString()).sortWith(_ < _)
    val exp = expected.collect().map(_.toString()).sortWith(_ < _)
    got.sameElements(exp)
  }

  /** Convert CoordinateMatrix to DenseMatrix, taken from Spark */
  protected def toBDM(matrix: CoordinateMatrix): BDM[Double] = {
    val m = matrix.numRows().toInt
    val n = matrix.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    matrix.entries.collect().foreach { case MatrixEntry(i, j, value) =>
      mat(i.toInt, j.toInt) = value
    }
    mat
  }

  /** Convert DenseMatrix into CoordinateMatrix */
  protected def toCoordinateMatrix(sc: SparkContext, matrix: BDM[Double]): CoordinateMatrix = {
    val buffer = ArrayBuffer[MatrixEntry]()
    matrix.foreachKey { case (i, j) =>
      buffer.append(MatrixEntry(i, j, matrix(i, j)))
    }
    new CoordinateMatrix(sc.parallelize(buffer.toSeq))
  }

  /** Compare CoordinateMatrix to expected DenseMatrix */
  protected def checkMatrixT(
      matrix: CoordinateMatrix,
      expected: BDM[Double],
      threshold: Double,
      abs: Boolean): Unit = {
    val localMatrix = toBDM(matrix)
    checkMatrixT(toBDM(matrix), expected, threshold, abs)
  }

  /** Compare mllib Matrix to expected DenseMatrix */
  protected def checkMatrixT(
      matrix: Matrix,
      expected: BDM[Double],
      threshold: Double,
      abs: Boolean): Unit = {
    val localMatrix = new BDM(matrix.numRows, matrix.numCols, matrix.toArray)
    checkMatrixT(localMatrix, expected, threshold, abs)
  }

  protected def checkMatrixT(
      matrix: BDM[Double],
      expected: BDM[Double],
      threshold: Double,
      abs: Boolean): Unit = {
    val msg = s"""
      > Matrix does not equal to expected matrix.
      >   Got:
      >$matrix
      >   Expected:
      >$expected
    """.stripMargin('>')
    require(matrix.rows == expected.rows && matrix.cols == expected.cols, msg)
    matrix.toArray.zip(expected.toArray).foreach { case (value1, value2) =>
      val delta = if (abs) Math.abs(value1) - Math.abs(value2) else value1 - value2
      require(Math.abs(delta) <= threshold, msg)
    }
  }

  protected def checkMatrix(matrix: CoordinateMatrix, expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, false)
  }

  protected def checkMatrix(matrix: Matrix, expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, false)
  }

  protected def checkMatrix(matrix: BDM[Double], expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, false)
  }

  protected def checkMatrixAbs(matrix: CoordinateMatrix, expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, true)
  }

  protected def checkMatrixAbs(matrix: Matrix, expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, true)
  }

  protected def checkMatrixAbs(matrix: BDM[Double], expected: BDM[Double]): Unit = {
    checkMatrixT(matrix, expected, 1e-8, true)
  }

  /** Compare distributed tensor to expected one */
  protected def checkTensor(tensor: DistributedTensor, expected: DistributedTensor): Unit = {
    if (tensor.numRows != expected.numRows || tensor.numCols != expected.numCols ||
        tensor.numLayers != expected.numLayers) {
      sys.error(s"Tensor dimensions of $tensor do not match expected $expected")
    }

    val entries = tensor.entries.collect.sortBy(_.hashCode).toSeq
    val expectedEntries = expected.entries.collect.sortBy(_.hashCode).toSeq

    val msg = s"""
    > Tensor does not equal expected tensor.
    >   Got:
    >${entries.mkString("[", ", ", "]")}
    >   Expected:
    >${expectedEntries.mkString("[", ", ", "]")}
    """.stripMargin('>')
    require(entries == expectedEntries, msg)
  }

  protected def checkTensorApproximate(
      tensor: DistributedTensor,
      expected: DistributedTensor,
      ignoreSign: Boolean = false): Unit = {
    val threshold = 0.001
    if (tensor.numRows != expected.numRows || tensor.numCols != expected.numCols ||
        tensor.numLayers != expected.numLayers) {
      sys.error(s"Tensor dimensions of $tensor do not match expected $expected")
    }

    val entries = tensor.entries.collect.sortBy(_.hashCode).toSeq
    val expectedEntries = expected.entries.collect.sortBy(_.hashCode).toSeq

    entries.foreach { entry =>
      val value1 = entry.value
      val value2 = expectedEntries.find { exp =>
        exp.i == entry.i && exp.j == entry.j && exp.k == entry.k
      }.get.value
      if (ignoreSign) {
        math.abs(value1) should be (math.abs(value2) +- threshold)
      } else {
        value1 should be (value2 +- threshold)
      }
    }
  }
}

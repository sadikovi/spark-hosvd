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

package com.github.sadikovi.hosvd.test

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.DataFrame

import com.github.sadikovi.hosvd.DistributedTensor

trait TestBase {
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
  protected def checkMatrix(matrix: CoordinateMatrix, expected: BDM[Double]): Unit = {
    val localMatrix = toBDM(matrix)
    val msg = s"""
    > Matrix does not equal expected matrix.
    >   Got:
    >$localMatrix
    >   Expected:
    >$expected
    """.stripMargin('>')
    require(localMatrix == expected, msg)
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
}

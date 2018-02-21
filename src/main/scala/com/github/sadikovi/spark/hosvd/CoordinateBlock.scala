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

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, Vectors}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

private[hosvd] case class ColumnValue(j: Long, value: Double)
private[hosvd] case class RowStripe(i: Long, cols: Seq[ColumnValue])

/**
 * Dataset based container for CoordinateMatrix.
 * Implements more efficient conversion methods.
 *
 * Code assumes that the max index within data is less than number of rows and number of columns.
 * In this case it is very similar to the Spark coordinate matrix.
 */
case class CoordinateBlock(
    @transient val data: Dataset[MatrixEntry],
    numRows: Long,
    numCols: Long) {
  import data.sparkSession.implicits._

  require(numRows >= 0 && numCols >= 0, s"Invalid dimensions, rows: $numRows, cols: $numCols")

  def toIndexedRowMatrix(): IndexedRowMatrix = {
    val n = numCols.toInt
    val rdd = data.select(col("i"), struct(col("j"), col("value")).as("column")).
      groupBy("i").agg(collect_list(col("column")).as("cols")).
      as[RowStripe].map { row =>
        val vector = Vectors.sparse(n, row.cols.map { cl => (cl.j.toInt, cl.value) })
        IndexedRow(row.i, vector)
      }.rdd
    new IndexedRowMatrix(rdd, numRows, n)
  }

  def toCoordinateMatrix(): CoordinateMatrix = {
    new CoordinateMatrix(data.rdd, numRows, numCols)
  }

  def toLocalMatrix(): Matrix = {
    val m = numRows.toInt
    val n = numCols.toInt
    val values = new Array[Double](m * n)
    data.collect.foreach { entry =>
      val indexOffset = entry.j.toInt * m + entry.i.toInt
      values(indexOffset) = entry.value
    }
    new DenseMatrix(m, n, values)
  }

  def transpose(): CoordinateBlock = {
    val ds = data.select(col("j").as("i"), col("i").as("j"), col("value")).as[MatrixEntry]
    CoordinateBlock(ds, numCols, numRows)
  }
}

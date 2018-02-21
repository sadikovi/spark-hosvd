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

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.sql.Dataset
import org.apache.spark.mllib.linalg.distributed.MatrixEntry

import com.github.sadikovi.spark.hosvd.test.{UnitTestSuite, SparkLocal}

class CoordinateBlockSuite extends UnitTestSuite with SparkLocal {
  override def beforeAll() {
    startSparkSession()
  }

  override def afterAll() {
    stopSparkSession()
  }

  test("CoordinateBlock - check dimensions") {
    val implicits = spark.implicits
    import implicits._

    val ds = Seq(MatrixEntry(0, 0, 1.0)).toDS
    var err = intercept[IllegalArgumentException] {
      CoordinateBlock(ds, -1, 0)
    }
    assert(err.getMessage.contains("Invalid dimensions"))

    err = intercept[IllegalArgumentException] {
      CoordinateBlock(ds, 0, -1)
    }
    assert(err.getMessage.contains("Invalid dimensions"))
  }

  test("CoordinateBlock - toLocalMatrix") {
    val implicits = spark.implicits
    import implicits._

    val ds = Seq(
      MatrixEntry(0, 0, 1.0),
      MatrixEntry(0, 1, 2.0),
      MatrixEntry(0, 2, 3.0),
      MatrixEntry(1, 0, 4.0),
      MatrixEntry(1, 1, 5.0),
      MatrixEntry(1, 2, 6.0)
    ).toDS

    val block = CoordinateBlock(ds, 2, 3)
    val mat = block.toLocalMatrix
    val exp = BDM(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0))
    checkMatrix(mat, exp)
  }

  test("CoordinateBlock - toIndexedRowMatrix") {
    val implicits = spark.implicits
    import implicits._

    val ds = Seq(
      MatrixEntry(0, 0, 1.0),
      MatrixEntry(0, 1, 2.0),
      MatrixEntry(0, 2, 3.0),
      MatrixEntry(1, 0, 4.0),
      MatrixEntry(1, 1, 5.0),
      MatrixEntry(1, 2, 6.0)
    ).toDS

    val block = CoordinateBlock(ds, 2, 3)
    val mat = block.
      toIndexedRowMatrix.
      toBlockMatrix.
      toLocalMatrix
    val exp = BDM(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0))
    checkMatrix(mat, exp)
  }

  test("CoordinateBlock - toCoordinateMatrix") {
    val implicits = spark.implicits
    import implicits._

    val ds = Seq(
      MatrixEntry(0, 0, 1.0),
      MatrixEntry(0, 1, 2.0),
      MatrixEntry(0, 2, 3.0),
      MatrixEntry(1, 0, 4.0),
      MatrixEntry(1, 1, 5.0),
      MatrixEntry(1, 2, 6.0)
    ).toDS

    val block = CoordinateBlock(ds, 2, 3)
    val mat = block.
      toCoordinateMatrix.
      toIndexedRowMatrix.
      toBlockMatrix.
      toLocalMatrix
    val exp = BDM(
      (1.0, 2.0, 3.0),
      (4.0, 5.0, 6.0))
    checkMatrix(mat, exp)
  }

  test("CoordinateBlock - transpose") {
    val implicits = spark.implicits
    import implicits._

    val ds = Seq(
      MatrixEntry(0, 0, 1.0),
      MatrixEntry(0, 1, 2.0),
      MatrixEntry(0, 2, 3.0),
      MatrixEntry(1, 0, 4.0),
      MatrixEntry(1, 1, 5.0),
      MatrixEntry(1, 2, 6.0)
    ).toDS

    val block = CoordinateBlock(ds, 2, 3).transpose
    val mat = block.toLocalMatrix
    val exp = BDM(
      (1.0, 4.0),
      (2.0, 5.0),
      (3.0, 6.0))
    checkMatrix(mat, exp)
  }
}

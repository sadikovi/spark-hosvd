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

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import com.github.sadikovi.spark.hosvd.test.{UnitTestSuite, SparkLocal}

class TensorSuite extends UnitTestSuite with SparkLocal {
  // test data for simple unfolding
  private var rdd: RDD[TensorEntry] = null

  override def beforeAll() {
    startSparkContext()
    rdd = sc.parallelize(
      TensorEntry(0, 0, 0, 0.0 + 111) ::
      TensorEntry(0, 1, 0, 10.0 + 111) ::
      TensorEntry(0, 2, 0, 20.0 + 111) ::
      TensorEntry(1, 0, 0, 100.0 + 111) ::
      TensorEntry(1, 1, 0, 110.0 + 111) ::
      TensorEntry(1, 2, 0, 120.0 + 111) ::
      TensorEntry(2, 0, 0, 200.0 + 111) ::
      TensorEntry(2, 1, 0, 210.0 + 111) ::
      TensorEntry(2, 2, 0, 220.0 + 111) ::
      TensorEntry(3, 0, 0, 300.0 + 111) ::
      TensorEntry(3, 1, 0, 310.0 + 111) ::
      TensorEntry(3, 2, 0, 320.0 + 111) ::
      TensorEntry(0, 0, 1, 1.0 + 111) ::
      TensorEntry(0, 1, 1, 11.0 + 111) ::
      TensorEntry(0, 2, 1, 21.0 + 111) ::
      TensorEntry(1, 0, 1, 101.0 + 111) ::
      TensorEntry(1, 1, 1, 111.0 + 111) ::
      TensorEntry(1, 2, 1, 121.0 + 111) ::
      TensorEntry(2, 0, 1, 201.0 + 111) ::
      TensorEntry(2, 1, 1, 211.0 + 111) ::
      TensorEntry(2, 2, 1, 221.0 + 111) ::
      TensorEntry(3, 0, 1, 301.0 + 111) ::
      TensorEntry(3, 1, 1, 311.0 + 111) ::
      TensorEntry(3, 2, 1, 321.0 + 111) ::
      Nil)
  }

  override def afterAll() {
    stopSparkContext()
  }

  test("TensorEntry - equals") {
    val entry = TensorEntry(0, 0, 0, 1.0)
    entry.equals(entry) should be (true)
    entry.equals(TensorEntry(0, 0, 0, 1.0)) should be (true)
    entry.equals(null) should be (false)
    entry.equals(TensorEntry(0, 0, 0, 1.1)) should be (false)
  }

  test("TensorEntry - hashCode") {
    val entry = TensorEntry(1, 2, 3, 4.5)
    entry.hashCode() should be (-290585598)
    TensorEntry(0, 0, 0, 0.0).hashCode should be (0)
  }

  test("TensorEntry - toString") {
    val entry = TensorEntry(1, 2, 3, 4.5)
    entry.toString should be ("[(1, 2, 3) -> 4.5]")
  }

  test("DistributedTensor - string repr") {
    val entries = sc.parallelize(TensorEntry(0, 0, 0, 1.0) :: Nil)
    val tensor = new DistributedTensor(entries, 4, 3, 2)
    tensor.toString should be ("DistributedTensor[4 x 3 x 2]")
  }

  test("Distributed tensor - use provided dimensions") {
    val entries = sc.parallelize(TensorEntry(0, 0, 0, 1.0) :: Nil)
    val tensor = new DistributedTensor(entries, 4, 3, 2)
    tensor.numRows should be (4)
    tensor.numCols should be (3)
    tensor.numLayers should be (2)
  }

  test("Distributed tensor - compute dimensions") {
    val entries = sc.parallelize(TensorEntry(0, 0, 0, 1.0) :: Nil)
    val tensor = new DistributedTensor(entries, 0, 0, 0)
    tensor.numRows should be (1)
    tensor.numCols should be (1)
    tensor.numLayers should be (1)
  }

  test("Distributed tensor - persist entries") {
    val entries = sc.parallelize(TensorEntry(0, 0, 0, 1.0) :: Nil)
    entries.getStorageLevel should be (StorageLevel.NONE)
    val tensor = new DistributedTensor(entries, 0, 0, 0)
    tensor.entries.getStorageLevel should be (StorageLevel.MEMORY_AND_DISK)
  }

  test("Distributed tensor - unfold A1") {
    val tensor = new DistributedTensor(rdd)
    val result = tensor.unfold(UnfoldDirection.A1)
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 121.0, 131.0, 112.0, 122.0, 132.0),
      (211.0, 221.0, 231.0, 212.0, 222.0, 232.0),
      (311.0, 321.0, 331.0, 312.0, 322.0, 332.0),
      (411.0, 421.0, 431.0, 412.0, 422.0, 432.0))
    result.direction should be (UnfoldDirection.A1)
    checkMatrix(result.matrix, expected)
  }

  test("Distributed tensor - unfold A2") {
    val tensor = new DistributedTensor(rdd)
    val result = tensor.unfold(UnfoldDirection.A2)
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 112.0, 211.0, 212.0, 311.0, 312.0, 411.0, 412.0),
      (121.0, 122.0, 221.0, 222.0, 321.0, 322.0, 421.0, 422.0),
      (131.0, 132.0, 231.0, 232.0, 331.0, 332.0, 431.0, 432.0))
    result.direction should be (UnfoldDirection.A2)
    checkMatrix(result.matrix, expected)
  }

  test("Distributed tensor - unfold A3") {
    val tensor = new DistributedTensor(rdd)
    val result = tensor.unfold(UnfoldDirection.A3)
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 211.0, 311.0, 411.0, 121.0, 221.0, 321.0, 421.0, 131.0, 231.0, 331.0, 431.0),
      (112.0, 212.0, 312.0, 412.0, 122.0, 222.0, 322.0, 422.0, 132.0, 232.0, 332.0, 432.0))
    result.direction should be (UnfoldDirection.A3)
    checkMatrix(result.matrix, expected)
  }

  test("Distributed tensor - invalid unfolding direction") {
    val tensor = new DistributedTensor(rdd)
    intercept[IllegalArgumentException] {
      tensor.unfold(null)
    }
  }

  test("Distributed tensor - failed dimensions for fold") {
    val matrix = toCoordinateMatrix(sc, BDM(
      (1.0, 2.0, 3.0),
      (1.0, 2.0, 3.0)))

    val exc1 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(matrix, UnfoldDirection.A1, 10, 6, 10)
    }
    exc1.getMessage.startsWith("Failed to match dimensions") should be (true)

    val exc2 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(matrix, UnfoldDirection.A2, 6, 10, 10)
    }
    exc2.getMessage.startsWith("Failed to match dimensions") should be (true)

    val exc3 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(matrix, UnfoldDirection.A3, 10, 10, 6)
    }
    exc3.getMessage.startsWith("Failed to match dimensions") should be (true)
  }

  test("Distributed tensor - fold A1") {
    val matrix = toCoordinateMatrix(sc, BDM(
      (111.0, 121.0, 131.0, 112.0, 122.0, 132.0),
      (211.0, 221.0, 231.0, 212.0, 222.0, 232.0),
      (311.0, 321.0, 331.0, 312.0, 322.0, 332.0),
      (411.0, 421.0, 431.0, 412.0, 422.0, 432.0)))
    val tensor = DistributedTensor.fold(matrix, UnfoldDirection.A1, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(rdd))
  }

  test("Distributed tensor - fold A2") {
    val matrix = toCoordinateMatrix(sc, BDM(
      (111.0, 112.0, 211.0, 212.0, 311.0, 312.0, 411.0, 412.0),
      (121.0, 122.0, 221.0, 222.0, 321.0, 322.0, 421.0, 422.0),
      (131.0, 132.0, 231.0, 232.0, 331.0, 332.0, 431.0, 432.0)))
    val tensor = DistributedTensor.fold(matrix, UnfoldDirection.A2, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(rdd))
  }

  test("Distributed tensor - fold A3") {
    val matrix = toCoordinateMatrix(sc, BDM(
      (111.0, 211.0, 311.0, 411.0, 121.0, 221.0, 321.0, 421.0, 131.0, 231.0, 331.0, 431.0),
      (112.0, 212.0, 312.0, 412.0, 122.0, 222.0, 322.0, 422.0, 132.0, 232.0, 332.0, 432.0)))
    val tensor = DistributedTensor.fold(matrix, UnfoldDirection.A3, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(rdd))
  }

  test("Distributed tensor - unfold -> fold (A1)") {
    val tensor = new DistributedTensor(rdd)
    val matrix = tensor.unfold(UnfoldDirection.A1).matrix
    val newTensor = DistributedTensor.fold(matrix, UnfoldDirection.A1, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - unfold -> fold (A2)") {
    val tensor = new DistributedTensor(rdd)
    val matrix = tensor.unfold(UnfoldDirection.A2).matrix
    val newTensor = DistributedTensor.fold(matrix, UnfoldDirection.A2, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - unfold -> fold (A3)") {
    val tensor = new DistributedTensor(rdd)
    val matrix = tensor.unfold(UnfoldDirection.A3).matrix
    val newTensor = DistributedTensor.fold(matrix, UnfoldDirection.A3, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - hosvd") {
    val tensor = new DistributedTensor(rdd)
    val ho = tensor.hosvd(2, 2, 2).asInstanceOf[DistributedTensor]
    val expected = new DistributedTensor(sc.parallelize(
      TensorEntry(0, 0, 0, -1438.912) ::
      TensorEntry(0, 0, 1, 0) ::
      TensorEntry(1, 0, 0, 0) ::
      TensorEntry(1, 0, 1, 0.931) ::
      TensorEntry(0, 1, 0, 0) ::
      TensorEntry(0, 1, 1, -0.058) ::
      TensorEntry(1, 1, 0, -15.226) ::
      TensorEntry(1, 1, 1, -0.048) :: Nil))
    checkTensorApproximate(ho, expected, ignoreSign = true)
  }
}

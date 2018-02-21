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

import breeze.linalg.{DenseMatrix => BDM, _}

import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.sql.Dataset
import org.apache.spark.storage.StorageLevel

import com.github.sadikovi.spark.hosvd.test.{UnitTestSuite, SparkLocal}

class TensorSuite extends UnitTestSuite with SparkLocal {
  // test data for simple unfolding
  private var data: Dataset[TensorEntry] = null

  override def beforeAll() {
    startSparkSession()
    val implicits = spark.implicits
    import implicits._

    data = Seq(
      TensorEntry(0, 0, 0, 0.0 + 111),
      TensorEntry(0, 1, 0, 10.0 + 111),
      TensorEntry(0, 2, 0, 20.0 + 111),
      TensorEntry(1, 0, 0, 100.0 + 111),
      TensorEntry(1, 1, 0, 110.0 + 111),
      TensorEntry(1, 2, 0, 120.0 + 111),
      TensorEntry(2, 0, 0, 200.0 + 111),
      TensorEntry(2, 1, 0, 210.0 + 111),
      TensorEntry(2, 2, 0, 220.0 + 111),
      TensorEntry(3, 0, 0, 300.0 + 111),
      TensorEntry(3, 1, 0, 310.0 + 111),
      TensorEntry(3, 2, 0, 320.0 + 111),
      TensorEntry(0, 0, 1, 1.0 + 111),
      TensorEntry(0, 1, 1, 11.0 + 111),
      TensorEntry(0, 2, 1, 21.0 + 111),
      TensorEntry(1, 0, 1, 101.0 + 111),
      TensorEntry(1, 1, 1, 111.0 + 111),
      TensorEntry(1, 2, 1, 121.0 + 111),
      TensorEntry(2, 0, 1, 201.0 + 111),
      TensorEntry(2, 1, 1, 211.0 + 111),
      TensorEntry(2, 2, 1, 221.0 + 111),
      TensorEntry(3, 0, 1, 301.0 + 111),
      TensorEntry(3, 1, 1, 311.0 + 111),
      TensorEntry(3, 2, 1, 321.0 + 111)
    ).toDS
  }

  override def afterAll() {
    stopSparkSession()
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
    val implicits = spark.implicits
    import implicits._

    val entries = Seq(TensorEntry(0, 0, 0, 1.0)).toDS
    val tensor = new DistributedTensor(entries, 4, 3, 2)
    tensor.toString should be ("DistributedTensor[4 x 3 x 2]")
  }

  test("Distributed tensor - use provided dimensions") {
    val implicits = spark.implicits
    import implicits._

    val entries = Seq(TensorEntry(0, 0, 0, 1.0)).toDS
    val tensor = new DistributedTensor(entries, 4, 3, 2)
    tensor.numRows should be (4)
    tensor.numCols should be (3)
    tensor.numLayers should be (2)
  }

  test("Distributed tensor - use provided dimensions 2") {
    val implicits = spark.implicits
    import implicits._

    val entries = Seq(TensorEntry(0, 0, 0, 1.0)).toDS
    val tensor = new DistributedTensor(entries, 0, 0, 0)
    tensor.numRows should be (0)
    tensor.numCols should be (0)
    tensor.numLayers should be (0)
  }

  test("Distributed tensor - compute dimensions") {
    val implicits = spark.implicits
    import implicits._

    val entries = Seq(TensorEntry(0, 0, 0, 1.0)).toDS
    val tensor = new DistributedTensor(entries, -1, -1, -1)
    tensor.numRows should be (1)
    tensor.numCols should be (1)
    tensor.numLayers should be (1)
  }

  test("Distributed tensor - persist entries") {
    val implicits = spark.implicits
    import implicits._

    val entries = Seq(TensorEntry(0, 0, 0, 1.0)).toDS
    val tensor = new DistributedTensor(entries, 0, 0, 0)
    tensor.tensorEntries.storageLevel should be (StorageLevel.NONE)
    tensor.persist()
    tensor.tensorEntries.storageLevel should be (StorageLevel.MEMORY_AND_DISK)
    tensor.unpersist()
  }

  test("Distributed tensor - unfold A1") {
    val tensor = new DistributedTensor(data)
    val result = tensor.unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult]
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 121.0, 131.0, 112.0, 122.0, 132.0),
      (211.0, 221.0, 231.0, 212.0, 222.0, 232.0),
      (311.0, 321.0, 331.0, 312.0, 322.0, 332.0),
      (411.0, 421.0, 431.0, 412.0, 422.0, 432.0))
    result.direction should be (UnfoldDirection.A1)
    checkMatrix(result.block.toLocalMatrix, expected)
  }

  test("Distributed tensor - unfold A2") {
    val tensor = new DistributedTensor(data)
    val result = tensor.unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult]
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 112.0, 211.0, 212.0, 311.0, 312.0, 411.0, 412.0),
      (121.0, 122.0, 221.0, 222.0, 321.0, 322.0, 421.0, 422.0),
      (131.0, 132.0, 231.0, 232.0, 331.0, 332.0, 431.0, 432.0))
    result.direction should be (UnfoldDirection.A2)
    checkMatrix(result.block.toLocalMatrix, expected)
  }

  test("Distributed tensor - unfold A3") {
    val tensor = new DistributedTensor(data)
    val result = tensor.unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult]
    result.isLocal should be (false)
    val expected = BDM(
      (111.0, 211.0, 311.0, 411.0, 121.0, 221.0, 321.0, 421.0, 131.0, 231.0, 331.0, 431.0),
      (112.0, 212.0, 312.0, 412.0, 122.0, 222.0, 322.0, 422.0, 132.0, 232.0, 332.0, 432.0))
    result.direction should be (UnfoldDirection.A3)
    checkMatrix(result.block.toLocalMatrix, expected)
  }

  test("Distributed tensor - invalid unfolding direction") {
    val tensor = new DistributedTensor(data)
    intercept[MatchError] {
      tensor.unfold(null)
    }
  }

  test("Distributed tensor - failed dimensions for fold") {
    val block = toCoordinateBlock(spark, BDM(
      (1.0, 2.0, 3.0),
      (1.0, 2.0, 3.0)))

    val exc1 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(block, UnfoldDirection.A1, 10, 6, 10)
    }
    exc1.getMessage.startsWith("Failed to match dimensions") should be (true)

    val exc2 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(block, UnfoldDirection.A2, 6, 10, 10)
    }
    exc2.getMessage.startsWith("Failed to match dimensions") should be (true)

    val exc3 = intercept[IllegalArgumentException] {
      DistributedTensor.fold(block, UnfoldDirection.A3, 10, 10, 6)
    }
    exc3.getMessage.startsWith("Failed to match dimensions") should be (true)
  }

  test("Distributed tensor - fold A1") {
    val block = toCoordinateBlock(spark, BDM(
      (111.0, 121.0, 131.0, 112.0, 122.0, 132.0),
      (211.0, 221.0, 231.0, 212.0, 222.0, 232.0),
      (311.0, 321.0, 331.0, 312.0, 322.0, 332.0),
      (411.0, 421.0, 431.0, 412.0, 422.0, 432.0)))
    val tensor = DistributedTensor.fold(block, UnfoldDirection.A1, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(data))
  }

  test("Distributed tensor - fold A2") {
    val block = toCoordinateBlock(spark, BDM(
      (111.0, 112.0, 211.0, 212.0, 311.0, 312.0, 411.0, 412.0),
      (121.0, 122.0, 221.0, 222.0, 321.0, 322.0, 421.0, 422.0),
      (131.0, 132.0, 231.0, 232.0, 331.0, 332.0, 431.0, 432.0)))
    val tensor = DistributedTensor.fold(block, UnfoldDirection.A2, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(data))
  }

  test("Distributed tensor - fold A3") {
    val block = toCoordinateBlock(spark, BDM(
      (111.0, 211.0, 311.0, 411.0, 121.0, 221.0, 321.0, 421.0, 131.0, 231.0, 331.0, 431.0),
      (112.0, 212.0, 312.0, 412.0, 122.0, 222.0, 322.0, 422.0, 132.0, 232.0, 332.0, 432.0)))
    val tensor = DistributedTensor.fold(block, UnfoldDirection.A3, 4, 3, 2)
    checkTensor(tensor.asInstanceOf[DistributedTensor], new DistributedTensor(data))
  }

  test("Distributed tensor - unfold -> fold (A1)") {
    val tensor = new DistributedTensor(data)
    val block = tensor.unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult].block
    val newTensor = DistributedTensor.fold(block, UnfoldDirection.A1, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - unfold -> fold (A2)") {
    val tensor = new DistributedTensor(data)
    val block = tensor.unfold(UnfoldDirection.A2).asInstanceOf[DistributedUnfoldResult].block
    val newTensor = DistributedTensor.fold(block, UnfoldDirection.A2, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - unfold -> fold (A3)") {
    val tensor = new DistributedTensor(data)
    val block = tensor.unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].block
    val newTensor = DistributedTensor.fold(block, UnfoldDirection.A3, tensor.numRows,
      tensor.numCols, tensor.numLayers).asInstanceOf[DistributedTensor]
    checkTensor(newTensor, tensor)
  }

  test("Distributed tensor - hosvd") {
    val implicits = spark.implicits
    import implicits._

    val tensor = new DistributedTensor(data)
    val ho = tensor.hosvd(2, 2, 2).coreTensor.asInstanceOf[DistributedTensor]
    val expected = new DistributedTensor(Seq(
      TensorEntry(0, 0, 0, -1438.912),
      TensorEntry(0, 0, 1, 0),
      TensorEntry(1, 0, 0, 0),
      TensorEntry(1, 0, 1, 0.931),
      TensorEntry(0, 1, 0, 0),
      TensorEntry(0, 1, 1, -0.058),
      TensorEntry(1, 1, 0, -15.226),
      TensorEntry(1, 1, 1, -0.048)).toDS)
    checkTensorApproximate(ho, expected, ignoreSign = true)
  }

  test("Distributed tensor - hosvd 2") {
    val implicits = spark.implicits
    import implicits._

    /*
    Matlab code:

    l1 = [
    0.5470, 0.1835, 0.9294, 0.3063;
    0.2963, 0.3685, 0.7757, 0.5085;
    0.7447, 0.6256, 0.4868, 0.5108;
    0.1890, 0.7802, 0.4359, 0.8176;
    0.6868, 0.0811, 0.4468, 0.7948;
    ];

    l2 = [
    0.6443, 0.9390, 0.2077, 0.1948;
    0.3786, 0.8759, 0.3012, 0.2259;
    0.8116, 0.5502, 0.4709, 0.1707;
    0.5328, 0.6225, 0.2305, 0.2277;
    0.3507, 0.5870, 0.8443, 0.4357;
    ];

    l3 = [
    0.3111, 0.9797, 0.5949, 0.1174;
    0.9234, 0.4389, 0.2622, 0.2967;
    0.4302, 0.1111, 0.6028, 0.3188;
    0.1848, 0.2581, 0.7112, 0.4242;
    0.9049, 0.4087, 0.2217, 0.5079;
    ];

    l4 = [
    0.0855, 0.7303, 0.9631, 0.6241;
    0.2625, 0.4886, 0.5468, 0.6791;
    0.8010, 0.5785, 0.5211, 0.3955;
    0.0292, 0.2373, 0.2316, 0.3674;
    0.9289, 0.4588, 0.4889, 0.9880;
    ];

    m = cat(3, l1, l2, l3, l4);
    c = hosvd(m, [3,3,3]);
    */

    // scalastyle:off
    val entries = Seq(
      TensorEntry(0, 0, 0, 0.5470), TensorEntry(0, 1, 0, 0.1835), TensorEntry(0, 2, 0, 0.9294), TensorEntry(0, 3, 0, 0.3063),
      TensorEntry(1, 0, 0, 0.2963), TensorEntry(1, 1, 0, 0.3685), TensorEntry(1, 2, 0, 0.7757), TensorEntry(1, 3, 0, 0.5085),
      TensorEntry(2, 0, 0, 0.7447), TensorEntry(2, 1, 0, 0.6256), TensorEntry(2, 2, 0, 0.4868), TensorEntry(2, 3, 0, 0.5108),
      TensorEntry(3, 0, 0, 0.1890), TensorEntry(3, 1, 0, 0.7802), TensorEntry(3, 2, 0, 0.4359), TensorEntry(3, 3, 0, 0.8176),
      TensorEntry(4, 0, 0, 0.6868), TensorEntry(4, 1, 0, 0.0811), TensorEntry(4, 2, 0, 0.4468), TensorEntry(4, 3, 0, 0.7948),

      TensorEntry(0, 0, 1, 0.6443), TensorEntry(0, 1, 1, 0.9390), TensorEntry(0, 2, 1, 0.2077), TensorEntry(0, 3, 1, 0.1948),
      TensorEntry(1, 0, 1, 0.3786), TensorEntry(1, 1, 1, 0.8759), TensorEntry(1, 2, 1, 0.3012), TensorEntry(1, 3, 1, 0.2259),
      TensorEntry(2, 0, 1, 0.8116), TensorEntry(2, 1, 1, 0.5502), TensorEntry(2, 2, 1, 0.4709), TensorEntry(2, 3, 1, 0.1707),
      TensorEntry(3, 0, 1, 0.5328), TensorEntry(3, 1, 1, 0.6225), TensorEntry(3, 2, 1, 0.2305), TensorEntry(3, 3, 1, 0.2277),
      TensorEntry(4, 0, 1, 0.3507), TensorEntry(4, 1, 1, 0.5870), TensorEntry(4, 2, 1, 0.8443), TensorEntry(4, 3, 1, 0.4357),

      TensorEntry(0, 0, 2, 0.3111), TensorEntry(0, 1, 2, 0.9797), TensorEntry(0, 2, 2, 0.5949), TensorEntry(0, 3, 2, 0.1174),
      TensorEntry(1, 0, 2, 0.9234), TensorEntry(1, 1, 2, 0.4389), TensorEntry(1, 2, 2, 0.2622), TensorEntry(1, 3, 2, 0.2967),
      TensorEntry(2, 0, 2, 0.4302), TensorEntry(2, 1, 2, 0.1111), TensorEntry(2, 2, 2, 0.6028), TensorEntry(2, 3, 2, 0.3188),
      TensorEntry(3, 0, 2, 0.1848), TensorEntry(3, 1, 2, 0.2581), TensorEntry(3, 2, 2, 0.7112), TensorEntry(3, 3, 2, 0.4242),
      TensorEntry(4, 0, 2, 0.9049), TensorEntry(4, 1, 2, 0.4087), TensorEntry(4, 2, 2, 0.2217), TensorEntry(4, 3, 2, 0.5079),

      TensorEntry(0, 0, 3, 0.0855), TensorEntry(0, 1, 3, 0.7303), TensorEntry(0, 2, 3, 0.9631), TensorEntry(0, 3, 3, 0.6241),
      TensorEntry(1, 0, 3, 0.2625), TensorEntry(1, 1, 3, 0.4886), TensorEntry(1, 2, 3, 0.5468), TensorEntry(1, 3, 3, 0.6791),
      TensorEntry(2, 0, 3, 0.8010), TensorEntry(2, 1, 3, 0.5785), TensorEntry(2, 2, 3, 0.5211), TensorEntry(2, 3, 3, 0.3955),
      TensorEntry(3, 0, 3, 0.0292), TensorEntry(3, 1, 3, 0.2373), TensorEntry(3, 2, 3, 0.2316), TensorEntry(3, 3, 3, 0.3674),
      TensorEntry(4, 0, 3, 0.9289), TensorEntry(4, 1, 3, 0.4588), TensorEntry(4, 2, 3, 0.4889), TensorEntry(4, 3, 3, 0.9880)
    ).toDS
    // scalastyle:on

    val tensor = new DistributedTensor(entries)
    val res = tensor.hosvd(3, 3, 3)
    val core = res.coreTensor

    // check core tensor
    checkMatrixAbs(core.getLayer(0), BDM(
      (-4.46399611738112, 0.0401457449506178, 0.0476955115397744),
      (0.0653100349599707, 0.493191023130688, -0.731872604065162),
      (-0.0768425671492504, -0.0169142559860555, -0.00705104400334135)
    ))

    checkMatrixAbs(core.getLayer(1), BDM(
      (0.0999212994929081, 0.481115476716687, 0.692488088239690),
      (0.0321330730148846, 0.606677868061273, 0.122390294672478),
      (-0.0561174474261208, -0.0743559447131433, -0.431054307745373)
    ))

    checkMatrixAbs(core.getLayer(2), BDM(
      (0.0572370577534474, -0.0763782788051653, 0.168214618855026),
      (-0.0254492600341017, 0.539793831216911, -0.211410567549553),
      (0.268158537616755, 0.506546738132916, 0.363850227887210)
    ))

    // check left singular vectors
    val u1 = res.leftSingularVectors(UnfoldDirection.A1)
    checkMatrixAbs(u1, BDM(
      (-0.485965038110453, -0.661363074504828, 0.404363991476708),
      (-0.441369339989566, -0.0520181309708688, 0.238566626233376),
      (-0.444846773918030, 0.170066428781170, -0.449239799718779),
      (-0.342724643627288, -0.231405876401135, -0.707148021787248),
      (-0.503668793788884, 0.690957110029604, 0.278748999775178)
    ))
    val u2 = res.leftSingularVectors(UnfoldDirection.A2)
    checkMatrixAbs(u2, BDM(
      (-0.512790267641945, -0.756073526466873, 0.298270390933075),
      (-0.519363411960773, -0.0508532352178608, -0.826647540066221),
      (-0.511686513062042, 0.614220144109088, 0.134571749658377),
      (-0.453304202582081, 0.220228030946190, 0.457798058743132)
    ))
    val u3 = res.leftSingularVectors(UnfoldDirection.A3)
    checkMatrixAbs(u3, BDM(
      (-0.523590727665897, -0.534456270626436, 0.360828450780812),
      (-0.467409170523832, 0.724259665241362, 0.502110625774689),
      (-0.459099881671587, 0.273986781418517, -0.776160785272608),
      (-0.544617954039027, -0.338726550037748, -0.123540212061815)
    ))

    // check singular values
    val sv1 = res.singularValues(UnfoldDirection.A1)
    checkMatrix(new DenseMatrix(sv1.size, 1, sv1.toArray), BDM(
      4.56671981421963,
      1.31738444605424,
      1.10352324584210
    ))
    val sv2 = res.singularValues(UnfoldDirection.A2)
    checkMatrix(new DenseMatrix(sv2.size, 1, sv2.toArray), BDM(
      4.51501646906673,
      1.37441197277310,
      1.26515219424008
    ))
    val sv3 = res.singularValues(UnfoldDirection.A3)
    checkMatrix(new DenseMatrix(sv3.size, 1, sv3.toArray), BDM(
      4.59467456668543,
      1.29046968861424,
      1.09099550530752
    ))
  }

  test("Distributed tensor - getLayer") {
    val tensor = new DistributedTensor(data)
    checkMatrix(tensor.getLayer(0), BDM(
      (111.0, 121.0, 131.0),
      (211.0, 221.0, 231.0),
      (311.0, 321.0, 331.0),
      (411.0, 421.0, 431.0)
    ))
    checkMatrix(tensor.getLayer(1), BDM(
      (112.0, 122.0, 132.0),
      (212.0, 222.0, 232.0),
      (312.0, 322.0, 332.0),
      (412.0, 422.0, 432.0)
    ))

    intercept[IllegalArgumentException] {
      tensor.getLayer(2)
    }
  }

  test("Distributed tensor - rand, hosvd") {
    val tensor = DistributedTensor.rand(spark, 10, 8, 4)
    val core = tensor.hosvd(7, 3, 2).coreTensor
    assert(core.numRows === 7)
    assert(core.numCols === 3)
    assert(core.numLayers === 2)
  }

  test("Distributed tensor - computeSVD") {
    val tensor = new DistributedTensor(data)
    val svd = tensor.computeSVD(4, UnfoldDirection.A1)
    assert(svd.U.numRows == 4 && svd.U.numCols == 4)
    assert(svd.V.numRows == 6 && svd.V.numCols == 4)
  }

  test("Distributed tensor - computeSVD, enable U or V") {
    import DistributedTensor._
    val tensor = rand(spark, 40, 7, 5)
    // transposed = false
    var block = tensor.unfold(UnfoldDirection.A1).asInstanceOf[DistributedUnfoldResult].block

    var svd = computeSVD(block, 5, StorageLevel.NONE, computeU = true, computeV = true)
    assert(svd.U != null && svd.V != null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = true, computeV = false)
    assert(svd.U != null && svd.V == null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = false, computeV = true)
    assert(svd.U == null && svd.V != null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = false, computeV = false)
    assert(svd.U == null && svd.V == null)

    // transposed = true
    block = tensor.unfold(UnfoldDirection.A3).asInstanceOf[DistributedUnfoldResult].block

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = true, computeV = true)
    assert(svd.U != null && svd.V != null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = true, computeV = false)
    assert(svd.U != null && svd.V == null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = false, computeV = true)
    assert(svd.U == null && svd.V != null)

    svd = computeSVD(block, 5, StorageLevel.NONE, computeU = false, computeV = false)
    assert(svd.U == null && svd.V == null)
  }

  test("DistributedTensor - multiply") {
    val implicits = spark.implicits
    import implicits._

    val a = Seq(
      MatrixEntry(0, 0, 0.8147),
      MatrixEntry(0, 1, 0.9134),
      MatrixEntry(1, 0, 0.9058),
      MatrixEntry(1, 1, 0.6324),
      MatrixEntry(2, 0, 0.1270),
      MatrixEntry(2, 1, 0.0975)
    ).toDS

    val b = Seq(
      MatrixEntry(0, 0, 0.2785),
      MatrixEntry(0, 1, 0.9575),
      MatrixEntry(0, 2, 0.1576),
      MatrixEntry(0, 3, 0.9572),
      MatrixEntry(1, 0, 0.5469),
      MatrixEntry(1, 1, 0.9649),
      MatrixEntry(1, 2, 0.9706),
      MatrixEntry(1, 3, 0.4854)
    ).toDS

    val blockA = CoordinateBlock(a, 3, 2)
    val blockB = CoordinateBlock(b, 2, 4)

    val res = DistributedTensor.multiply(blockA, blockB.toLocalMatrix).toLocalMatrix
    val exp = BDM(
      (0.7264, 1.6614, 1.0149, 1.2232),
      (0.5981, 1.4775, 0.7566, 1.1740),
      (0.0887, 0.2157, 0.1146, 0.1689)
    )
    checkMatrixT(res, exp, 1e-4, false)
  }
}

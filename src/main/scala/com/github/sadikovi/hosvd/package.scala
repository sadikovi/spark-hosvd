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

package com.github.sadikovi.hosvd

import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix

/** Direction to unfold tensor */
object UnfoldDirection extends Enumeration {
  type UnfoldDirection = Value
  val A1, A2, A3 = Value
}

/** Abstract class for result of unfolding */
abstract class UnfoldResult {
  /** Whether or not result is local matrix */
  def isLocal: Boolean

  /** Direction used for this unfolding */
  def direction: UnfoldDirection.Value
}

/** Unfold result for [[DistributedTensor]] */
private[hosvd] case class DistributedUnfoldResult(
    @transient matrix: CoordinateMatrix,
    private val unfoldDirection: UnfoldDirection.Value)
  extends UnfoldResult {

  override def isLocal: Boolean = false

  override def direction: UnfoldDirection.Value = unfoldDirection
}

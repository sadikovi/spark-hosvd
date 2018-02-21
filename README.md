# spark-hosvd
Spark High Order SVD

[![Build Status](https://travis-ci.org/sadikovi/spark-hosvd.svg?branch=master)](https://travis-ci.org/sadikovi/spark-hosvd)
[![Coverage Status](https://coveralls.io/repos/github/sadikovi/spark-hosvd/badge.svg?branch=master)](https://coveralls.io/github/sadikovi/spark-hosvd?branch=master)

## Overview
Small library for Apache Spark to compute unfolding/folding/HOSVD for distributed 3rd order tensor.

## Requirements
| Spark version | spark-hosvd latest version |
|---------------|------------------------------|
| 1.6.x | 0.1.0 |
| 2.2.x | 0.2.0 |

## Building From Source
This library is built using `sbt`, to build a JAR file simply run `sbt package` from project root.

## Testing
Run `sbt test` from project root.

## Usage
Import jar into spark-shell or add dependency in main jar for spark-submit.
Use `import com.github.sadikovi.spark.hosvd._` to import all building blocks for tensor.
I recommend to check out available methods in the code and tests for different usage.

## Example

### Scala API

#### Unfolding tensor
```scala
import com.github.sadikovi.spark.hosvd._

val entries: Dataset[TensorEntry] = ...
val tensor = new DistributedTensor(entries, 4, 3, 2)
val result = tensor.unfold(UnfoldDirection.A1)
val matrix: CoordinateMatrix =
  result.asInstanceOf[DistributedUnfoldResult].block.toCoordinateMatrix
```

#### Folding matrix
Tensor folding by providing unfold direction and actual tensor dimensions to fold into.
```scala
val matrix: CoordinateMatrix = ...
val block = CoordinateBlock(matrix.entries.toDS, matrix.numRows, matrix.numCols)
val tensor = DistributedTensor.fold(block, UnfoldDirection.A1, 4, 3, 2)
```

#### HOSVD
High-order SVD by providing desired dimensions for core tensor (singular values to keep).
```scala
val tensor = new DistributedTensor(entries, 4, 3, 2)
val result = tensor.hosvd(2, 2, 2)
val core = result.coreTensor
```

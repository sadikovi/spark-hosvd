name := "spark-hosvd"

organization := "com.github.sadikovi"

scalaVersion := "2.11.7"

spName := "sadikovi/spark-hosvd"

val defaultSparkVersion = "2.2.1"

sparkVersion := sys.props.getOrElse("spark.testVersion", defaultSparkVersion)

val defaultHadoopVersion = "2.6.0"

val hadoopVersion = settingKey[String]("The version of Hadoop to test against.")

hadoopVersion := sys.props.getOrElse("hadoop.testVersion", defaultHadoopVersion)

spAppendScalaVersion := true

spIncludeMaven := false

spIgnoreProvided := true

sparkComponents := Seq("mllib")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.4" % "test"
)

libraryDependencies ++= Seq(
  "org.apache.hadoop" % "hadoop-client" % hadoopVersion.value % "test" exclude("javax.servlet", "servlet-api") force(),
  "org.apache.spark" %% "spark-core" % sparkVersion.value % "test" exclude("org.apache.hadoop", "hadoop-client"),
  "org.apache.spark" %% "spark-mllib" % sparkVersion.value % "test" exclude("org.apache.hadoop", "hadoop-client")
)

// check deprecation without manual restart
scalacOptions in ThisBuild ++= Seq("-unchecked", "-deprecation", "-feature")

// Display full-length stacktraces from ScalaTest
testOptions in Test += Tests.Argument("-oF")

parallelExecution in Test := false
fork in Test := true // to run tests in a separate JVM
javaOptions += "-Xmx4G" // increase maximum heap size for forked processes

// Skip tests during assembly
test in assembly := {}

coverageHighlighting := {
  if (scalaBinaryVersion.value == "2.10") false
  else true
}
coverageMinimum := 80
coverageFailOnMinimum := true

EclipseKeys.eclipseOutput := Some("target/eclipse")

// tasks dependencies
lazy val compileScalastyle = taskKey[Unit]("compileScalastyle")
compileScalastyle := org.scalastyle.sbt.ScalastylePlugin.scalastyle.in(Compile).toTask("").value
(compile in Compile) <<= (compile in Compile).dependsOn(compileScalastyle)

// Create a default Scala style task to run with tests
lazy val testScalastyle = taskKey[Unit]("testScalastyle")
testScalastyle := org.scalastyle.sbt.ScalastylePlugin.scalastyle.in(Test).toTask("").value
(test in Test) <<= (test in Test).dependsOn(testScalastyle)

/********************
 * Release settings *
 ********************/

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (version.value.endsWith("SNAPSHOT"))
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishMavenStyle := true

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

releaseCrossBuild := true

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

releasePublishArtifactsAction := PgpKeys.publishSigned.value

pomExtra := (
  <url>https://github.com/sadikovi/spark-hosvd</url>
  <scm>
    <url>git@github.com:sadikovi/spark-hosvd.git</url>
    <connection>scm:git:git@github.com:sadikovi/spark-hosvd.git</connection>
  </scm>
  <developers>
    <developer>
      <id>sadikovi</id>
      <name>Ivan Sadikov</name>
      <url>https://github.com/sadikovi</url>
    </developer>
  </developers>
)

bintrayReleaseOnPublish in ThisBuild := false

import ReleaseTransformations._

// Add publishing to spark packages as another step.
releaseProcess := Seq[ReleaseStep](
  checkSnapshotDependencies,
  inquireVersions,
  runTest,
  setReleaseVersion,
  commitReleaseVersion,
  tagRelease,
  publishArtifacts,
  setNextVersion,
  commitNextVersion,
  pushChanges,
  releaseStepTask(spPublish)
)

// Credentials for sbt-spark-package
credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")
// Credentials for publishing to sonatype
credentials += Credentials(Path.userHome / ".ivy2" / ".sonatype.sbt")

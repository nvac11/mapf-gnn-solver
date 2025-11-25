package cbs

import cbs.utils.DataGenerator
object App {
  def main(args: Array[String]): Unit = {
    DataGenerator.generateDataset(
      outputFile = "cbs_dataset_large.json",
      targetSamples = 100_000, 
      batchSize = 2000
    )
  }
}
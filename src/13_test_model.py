from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel

spark = SparkSession.builder \
    .appName("TestSavedRandomForest") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

model_path = "hdfs://namenode:9000/smart-home/historical/models/random_forest_final"

model = RandomForestRegressionModel.load(model_path)

print("=" * 60)
print("MODEL LOADED SUCCESSFULLY")
print("=" * 60)
print("Number of trees:", model.getNumTrees)
print("Max depth:", model.getOrDefault(model.maxDepth))
print("=" * 60)

spark.stop()
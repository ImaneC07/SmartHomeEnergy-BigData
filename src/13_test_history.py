from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("TestHistoricalStreamingFeatures") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

historical_path = (
    "hdfs://namenode:9000/"
    "smart-home/historical/features/"
    "household_power_consumption_features"
)

df = spark.read.parquet(historical_path)

print("=" * 70)
print("HISTORICAL DATA LOADED")
print("=" * 70)

print("Number of rows:", df.count())

print("\nLatest 10 records:")
df.orderBy(col("timestamp").desc()) \
    .select(
        "timestamp",
        "Global_active_power",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "lag_24",
        "rolling_mean_3h",
        "rolling_mean_6h"
    ) \
    .show(10, truncate=False)

print("=" * 70)

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json,
    col,
    to_timestamp,
    hour,
    dayofmonth,
    dayofweek,
    month,
    when,
    lit
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType
)
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler


# ============================================================
# 1. Spark
# ============================================================

spark = SparkSession.builder \
    .appName("SmartHomeEnergyStreamingPrediction") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 2. Load saved Random Forest model
# ============================================================

model_path = (
    "hdfs://namenode:9000/"
    "smart-home/historical/models/random_forest_final"
)

model = RandomForestRegressionModel.load(model_path)

print("=" * 70)
print("RANDOM FOREST MODEL LOADED")
print("=" * 70)
print("Number of trees:", model.getNumTrees)
print("Max depth:", model.getOrDefault(model.maxDepth))
print("=" * 70)


# ============================================================
# 3. Kafka JSON schema
# ============================================================

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("Global_active_power", DoubleType(), True),
    StructField("Global_reactive_power", DoubleType(), True),
    StructField("Voltage", DoubleType(), True),
    StructField("Global_intensity", DoubleType(), True),
    StructField("Sub_metering_1", DoubleType(), True),
    StructField("Sub_metering_2", DoubleType(), True),
    StructField("Sub_metering_3", DoubleType(), True)
])


# ============================================================
# 4. Read Kafka
# ============================================================

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "smart-home-energy") \
    .option("startingOffsets", "latest") \
    .load()


# ============================================================
# 5. Parse JSON
# ============================================================

json_df = kafka_df.select(
    from_json(
        col("value").cast("string"),
        schema
    ).alias("data")
)

stream_df = json_df.select("data.*")

stream_df = stream_df.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
)


# ============================================================
# 6. Temporary feature construction
# ============================================================
#
# IMPORTANT:
# The trained model requires 22 features.
#
# For this first streaming test, we create the feature columns
# from the incoming measurements so that we can verify the
# complete Kafka -> Spark -> ML pipeline.
#
# We will replace these temporary values with real historical/
# state-based lag and rolling features in the next step.
# ============================================================

features_df = stream_df \
    .withColumn("lag_1", col("Global_active_power")) \
    .withColumn("lag_2", col("Global_active_power")) \
    .withColumn("lag_3", col("Global_active_power")) \
    .withColumn("lag_4", col("Global_active_power")) \
    .withColumn("lag_24", col("Global_active_power")) \
    .withColumn("lag_voltage_1", col("Voltage")) \
    .withColumn("lag_intensity_1", col("Global_intensity")) \
    .withColumn("rolling_mean_3h", col("Global_active_power")) \
    .withColumn("rolling_std_3h", lit(0.0)) \
    .withColumn("rolling_mean_6h", col("Global_active_power")) \
    .withColumn("hour", hour(col("timestamp"))) \
    .withColumn("day", dayofmonth(col("timestamp"))) \
    .withColumn("day_of_week", dayofweek(col("timestamp"))) \
    .withColumn("month", month(col("timestamp"))) \
    .withColumn(
        "is_weekend",
        when(dayofweek(col("timestamp")).isin([1, 7]), 1)
        .otherwise(0)
    )


# ============================================================
# 7. Assemble the SAME 22 features used during training
# ============================================================

feature_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "lag_24",
    "lag_voltage_1",
    "lag_intensity_1",
    "rolling_mean_3h",
    "rolling_std_3h",
    "rolling_mean_6h",
    "hour",
    "day",
    "day_of_week",
    "month",
    "is_weekend"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

prediction_df = assembler.transform(features_df)

prediction_df = model.transform(prediction_df)


# ============================================================
# 8. Display prediction
# ============================================================

result_df = prediction_df.select(
    "timestamp",
    "Global_active_power",
    "prediction"
)

query = result_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .option("numRows", 20) \
    .start()

query.awaitTermination()
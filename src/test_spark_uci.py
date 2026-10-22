from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TestSparkUCI") \
    .getOrCreate()

df = spark.read \
    .option("header", "true") \
    .option("sep", ";") \
    .option("inferSchema", "true") \
    .csv("hdfs://namenode:9000/smart-home/historical/raw/household_power_consumption.txt")

print("=== Premières lignes du dataset UCI ===")
df.show(10, truncate=False)

print("=== Schéma ===")
df.printSchema()

print("=== Nombre de lignes ===")
print(df.count())

spark.stop()

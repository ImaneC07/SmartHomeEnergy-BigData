from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TestSparkHDFS") \
    .getOrCreate()

df = spark.read.json(
    "hdfs://namenode:9000/smart-home/raw/kafka_raw.json"
)

print("=== Données lues depuis HDFS ===")
df.show()

print("=== Schéma ===")
df.printSchema()

spark.stop()
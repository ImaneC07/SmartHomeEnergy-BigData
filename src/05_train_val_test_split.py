from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime

spark = SparkSession.builder \
    .appName("TrainValTestSplit") \
    .getOrCreate()

INPUT_PATH = "hdfs://namenode:9000/smart-home/historical/features/household_power_consumption_features"

TRAIN_PATH = "hdfs://namenode:9000/smart-home/historical/ml/train"
VAL_PATH = "hdfs://namenode:9000/smart-home/historical/ml/validation"
TEST_PATH = "hdfs://namenode:9000/smart-home/historical/ml/test"

print("=== Lecture des features ===")

df = spark.read.parquet(INPUT_PATH)

print(f"Nombre total de lignes : {df.count()}")

# ============================================================
# 1. Conversion temporaire du timestamp en secondes
# ============================================================

print("=== Conversion du timestamp ===")

df = df.withColumn(
    "timestamp_unix",
    col("timestamp").cast("long")
)

# ============================================================
# 2. Détermination des dates de séparation
# ============================================================

print("=== Détermination des dates de split ===")

quantiles = df.approxQuantile(
    "timestamp_unix",
    [0.70, 0.85],
    0.001
)

train_end = quantiles[0]
val_end = quantiles[1]

print(f"TRAIN timestamp Unix : {train_end}")
print(f"VALIDATION timestamp Unix : {val_end}")

# ============================================================
# 3. Création des datasets
# ============================================================

print("=== Création du TRAIN ===")

train_df = df.filter(
    col("timestamp_unix") <= train_end
).drop("timestamp_unix")

print("=== Création de la VALIDATION ===")

val_df = df.filter(
    (col("timestamp_unix") > train_end) &
    (col("timestamp_unix") <= val_end)
).drop("timestamp_unix")

print("=== Création du TEST ===")

test_df = df.filter(
    col("timestamp_unix") > val_end
).drop("timestamp_unix")

# ============================================================
# 4. Vérification des tailles
# ============================================================

train_count = train_df.count()
val_count = val_df.count()
test_count = test_df.count()

print("=== Résultats du split ===")

print(f"TRAIN      : {train_count}")
print(f"VALIDATION : {val_count}")
print(f"TEST       : {test_count}")

print(f"TOTAL      : {train_count + val_count + test_count}")

# ============================================================
# 5. Vérification des périodes
# ============================================================

print("=== Période TRAIN ===")

train_df.select("timestamp") \
    .orderBy("timestamp") \
    .show(1, truncate=False)

print("=== Période VALIDATION ===")

val_df.select("timestamp") \
    .orderBy("timestamp") \
    .show(1, truncate=False)

print("=== Période TEST ===")

test_df.select("timestamp") \
    .orderBy("timestamp") \
    .show(1, truncate=False)

# ============================================================
# 6. Sauvegarde dans HDFS
# ============================================================

print("=== Sauvegarde TRAIN ===")

train_df.write \
    .mode("overwrite") \
    .parquet(TRAIN_PATH)

print("=== Sauvegarde VALIDATION ===")

val_df.write \
    .mode("overwrite") \
    .parquet(VAL_PATH)

print("=== Sauvegarde TEST ===")

test_df.write \
    .mode("overwrite") \
    .parquet(TEST_PATH)

print("=== Split terminé avec succès ===")

spark.stop()
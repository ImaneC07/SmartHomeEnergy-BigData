from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder \
    .appName("PrepareMLData") \
    .getOrCreate()

TRAIN_INPUT = "hdfs://namenode:9000/smart-home/historical/ml/train"
VAL_INPUT = "hdfs://namenode:9000/smart-home/historical/ml/validation"
TEST_INPUT = "hdfs://namenode:9000/smart-home/historical/ml/test"

TRAIN_OUTPUT = "hdfs://namenode:9000/smart-home/historical/ml_prepared/train"
VAL_OUTPUT = "hdfs://namenode:9000/smart-home/historical/ml_prepared/validation"
TEST_OUTPUT = "hdfs://namenode:9000/smart-home/historical/ml_prepared/test"

print("=== Lecture des datasets ===")

train_df = spark.read.parquet(TRAIN_INPUT)
val_df = spark.read.parquet(VAL_INPUT)
test_df = spark.read.parquet(TEST_INPUT)

print(f"TRAIN : {train_df.count()}")
print(f"VALIDATION : {val_df.count()}")
print(f"TEST : {test_df.count()}")

# ============================================================
# Features utilisées par le modèle
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

print("=== Features utilisées ===")

for feature in feature_columns:
    print(f"- {feature}")

# ============================================================
# VectorAssembler
# ============================================================

print("=== Création du vecteur de features ===")

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features"
)

train_ml = assembler.transform(train_df)
val_ml = assembler.transform(val_df)
test_ml = assembler.transform(test_df)

# ============================================================
# Sélection finale
# ============================================================

train_ml = train_ml.select(
    "timestamp",
    "features",
    "target"
)

val_ml = val_ml.select(
    "timestamp",
    "features",
    "target"
)

test_ml = test_ml.select(
    "timestamp",
    "features",
    "target"
)

# ============================================================
# Vérification
# ============================================================

print("=== Aperçu TRAIN préparé ===")
train_ml.show(5, truncate=False)

print("=== Schéma TRAIN préparé ===")
train_ml.printSchema()

print("=== Nombre de lignes ===")
print(f"TRAIN : {train_ml.count()}")
print(f"VALIDATION : {val_ml.count()}")
print(f"TEST : {test_ml.count()}")

# ============================================================
# Sauvegarde
# ============================================================

print("=== Sauvegarde TRAIN ===")
train_ml.write.mode("overwrite").parquet(TRAIN_OUTPUT)

print("=== Sauvegarde VALIDATION ===")
val_ml.write.mode("overwrite").parquet(VAL_OUTPUT)

print("=== Sauvegarde TEST ===")
test_ml.write.mode("overwrite").parquet(TEST_OUTPUT)

print("=== Préparation ML terminée avec succès ===")

spark.stop()
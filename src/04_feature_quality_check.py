from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, min, max

spark = SparkSession.builder \
    .appName("FeatureQualityCheck") \
    .getOrCreate()

INPUT_PATH = "hdfs://namenode:9000/smart-home/historical/features/household_power_consumption_features"

print("=== Lecture du dataset avec features ===")

df = spark.read.parquet(INPUT_PATH)

# ============================================================
# 1. Nombre de lignes
# ============================================================

print("=== Nombre de lignes ===")

row_count = df.count()

print(f"Nombre de lignes avec features : {row_count}")

# ============================================================
# 2. Schéma
# ============================================================

print("=== Schéma ===")

df.printSchema()

# ============================================================
# 3. Aperçu
# ============================================================

print("=== Aperçu des données ===")

df.show(10, truncate=False)

# ============================================================
# 4. Vérification des NULL
# ============================================================

print("=== Vérification des valeurs NULL ===")

null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

null_counts.show()

# ============================================================
# 5. Période des données
# ============================================================

print("=== Période des données ===")

df.select(
    min("timestamp").alias("date_min"),
    max("timestamp").alias("date_max")
).show()

# ============================================================
# 6. Vérification des statistiques de target
# ============================================================

print("=== Statistiques de la target ===")

df.select(
    "target"
).describe().show()

# ============================================================
# 7. Vérification des colonnes importantes
# ============================================================

expected_columns = [
    "target",
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

print("=== Vérification des features attendues ===")

missing_columns = [
    c for c in expected_columns
    if c not in df.columns
]

if missing_columns:
    print(f"ERREUR : colonnes manquantes : {missing_columns}")
else:
    print("Toutes les features attendues sont présentes.")

spark.stop()

print("=== Feature Quality Check terminé avec succès ===")
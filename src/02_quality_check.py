from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, min, max, when


# ============================================================
# 1. Création de la SparkSession
# ============================================================

spark = SparkSession.builder \
    .appName("QualityCheckHistorical") \
    .getOrCreate()


# ============================================================
# 2. Chemin du dataset nettoyé
# ============================================================

INPUT_PATH = "hdfs://namenode:9000/smart-home/historical/processed/household_power_consumption_clean"


# ============================================================
# 3. Lecture du Parquet
# ============================================================

print("=== Lecture du dataset nettoyé ===")

df = spark.read.parquet(INPUT_PATH)


# ============================================================
# 4. Nombre de lignes
# ============================================================

print("=== Nombre de lignes ===")

row_count = df.count()

print(f"Nombre de lignes nettoyées : {row_count}")


# ============================================================
# 5. Schéma
# ============================================================

print("=== Schéma ===")

df.printSchema()


# ============================================================
# 6. Aperçu des données
# ============================================================

print("=== Aperçu des données ===")

df.show(10, truncate=False)


# ============================================================
# 7. Vérification des valeurs NULL
# ============================================================

print("=== Vérification des valeurs NULL ===")

null_counts = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

null_counts.show()


# ============================================================
# 8. Période couverte par les données
# ============================================================

print("=== Période des données ===")

df.select(
    min("timestamp").alias("date_min"),
    max("timestamp").alias("date_max")
).show()


# ============================================================
# 9. Fin
# ============================================================

spark.stop()

print("=== Quality Check terminé avec succès ===")
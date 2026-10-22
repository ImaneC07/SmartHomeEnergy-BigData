from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, concat_ws, to_timestamp


# ============================================================
# 1. Création de la SparkSession
# ============================================================

spark = SparkSession.builder \
    .appName("CleanHistoricalUCI") \
    .getOrCreate()


# ============================================================
# 2. Chemins HDFS
# ============================================================

INPUT_PATH = "hdfs://namenode:9000/smart-home/historical/raw/household_power_consumption.txt"

OUTPUT_PATH = "hdfs://namenode:9000/smart-home/historical/processed/household_power_consumption_clean"


# ============================================================
# 3. Lecture du dataset brut
# ============================================================

print("=== Lecture du dataset UCI depuis HDFS ===")

df = spark.read \
    .option("header", "true") \
    .option("sep", ";") \
    .csv(INPUT_PATH)

print(f"Nombre de lignes brutes : {df.count()}")


# ============================================================
# 4. Remplacement des valeurs '?' par NULL
# ============================================================

print("=== Nettoyage des valeurs manquantes ===")

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column_name in numeric_columns:
    df = df.withColumn(
        column_name,
        when(col(column_name) == "?", None)
        .otherwise(col(column_name))
    )


# ============================================================
# 5. Création correcte du timestamp
# ============================================================

print("=== Création du timestamp ===")

df = df.withColumn(
    "timestamp",
    to_timestamp(
        concat_ws(" ", col("Date"), col("Time")),
        "d/M/yyyy HH:mm:ss"
    )
)


# ============================================================
# 6. Conversion des colonnes numériques en Double
# ============================================================

print("=== Conversion des colonnes numériques ===")

for column_name in numeric_columns:
    df = df.withColumn(
        column_name,
        col(column_name).cast("double")
    )


# ============================================================
# 7. Sélection des colonnes utiles
# ============================================================

df = df.select(
    "timestamp",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
)


# ============================================================
# 8. Suppression des lignes avec valeurs manquantes
# ============================================================

print("=== Suppression des lignes incomplètes ===")

df_clean = df.dropna()

print(f"Nombre de lignes après nettoyage : {df_clean.count()}")


# ============================================================
# 9. Vérification
# ============================================================

print("=== Aperçu des données nettoyées ===")

df_clean.show(10, truncate=False)

print("=== Schéma final ===")

df_clean.printSchema()


# ============================================================
# 10. Sauvegarde en Parquet dans HDFS
# ============================================================

print("=== Sauvegarde dans HDFS ===")

df_clean.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)

print(f"Données nettoyées sauvegardées dans : {OUTPUT_PATH}")


# ============================================================
# 11. Fin
# ============================================================

spark.stop()

print("=== Nettoyage terminé avec succès ===")

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    avg,
    stddev,
    hour,
    dayofmonth,
    dayofweek,
    month,
    when
)
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("FeatureEngineeringHistorical") \
    .getOrCreate()

INPUT_PATH = "hdfs://namenode:9000/smart-home/historical/processed/household_power_consumption_clean"

OUTPUT_PATH = "hdfs://namenode:9000/smart-home/historical/features/household_power_consumption_features"

print("=== Lecture du dataset nettoyé ===")

df = spark.read.parquet(INPUT_PATH)

print(f"Nombre de lignes initiales : {df.count()}")

# ============================================================
# 1. Définition de la fenêtre temporelle
# ============================================================

print("=== Création de la fenêtre temporelle ===")

window = Window.orderBy("timestamp")

# ============================================================
# 2. Création de la cible
# ============================================================

print("=== Création de la variable target ===")

df = df.withColumn(
    "target",
    lag("Global_active_power", -1).over(window)
)

# ============================================================
# 3. Variables retardées (lags)
# ============================================================

print("=== Création des variables lag ===")

df = df.withColumn("lag_1", lag("Global_active_power", 1).over(window))
df = df.withColumn("lag_2", lag("Global_active_power", 2).over(window))
df = df.withColumn("lag_3", lag("Global_active_power", 3).over(window))
df = df.withColumn("lag_4", lag("Global_active_power", 4).over(window))

# Environ 24 heures avant (1440 minutes)
df = df.withColumn(
    "lag_24",
    lag("Global_active_power", 1440).over(window)
)

# ============================================================
# 4. Lags d'autres variables
# ============================================================

print("=== Création des lags supplémentaires ===")

df = df.withColumn(
    "lag_voltage_1",
    lag("Voltage", 1).over(window)
)

df = df.withColumn(
    "lag_intensity_1",
    lag("Global_intensity", 1).over(window)
)

# ============================================================
# 5. Moyennes mobiles
# ============================================================

print("=== Création des moyennes mobiles ===")

window_3h = Window \
    .orderBy("timestamp") \
    .rowsBetween(-179, 0)

window_6h = Window \
    .orderBy("timestamp") \
    .rowsBetween(-359, 0)

df = df.withColumn(
    "rolling_mean_3h",
    avg("Global_active_power").over(window_3h)
)

df = df.withColumn(
    "rolling_mean_6h",
    avg("Global_active_power").over(window_6h)
)

# ============================================================
# 6. Écart-type mobile
# ============================================================

print("=== Création de rolling_std_3h ===")

df = df.withColumn(
    "rolling_std_3h",
    stddev("Global_active_power").over(window_3h)
)

# ============================================================
# 7. Variables temporelles
# ============================================================

print("=== Création des variables temporelles ===")

df = df.withColumn("hour", hour("timestamp"))
df = df.withColumn("day", dayofmonth("timestamp"))
df = df.withColumn("day_of_week", dayofweek("timestamp"))
df = df.withColumn("month", month("timestamp"))

df = df.withColumn(
    "is_weekend",
    when(col("day_of_week").isin(1, 7), 1).otherwise(0)
)

# ============================================================
# 8. Sélection finale
# ============================================================

print("=== Sélection des features finales ===")

df_features = df.select(
    "timestamp",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
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
)

# ============================================================
# 9. Suppression des lignes incomplètes
# ============================================================

print("=== Suppression des lignes avec features incomplètes ===")

df_features = df_features.dropna()

print(
    f"Nombre de lignes après Feature Engineering : "
    f"{df_features.count()}"
)

# ============================================================
# 10. Aperçu
# ============================================================

print("=== Aperçu des features ===")

df_features.show(10, truncate=False)

print("=== Schéma final ===")

df_features.printSchema()

# ============================================================
# 11. Sauvegarde HDFS
# ============================================================

print("=== Sauvegarde des features dans HDFS ===")

df_features.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)

print(f"Features sauvegardées dans : {OUTPUT_PATH}")

spark.stop()

print("=== Feature Engineering terminé avec succès ===")
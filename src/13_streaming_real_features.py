import os
import sys
from collections import deque
from datetime import datetime
import statistics

from pymongo import MongoClient
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)


# ============================================================
# 1. Spark Session
# ============================================================

spark = SparkSession.builder \
    .appName("SmartHomeEnergy-ContinuousStreaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# MongoDB Connection (Auto-détection localhost / mongodb container)
# ============================================================

def get_mongo_collection():
    uris = [
        os.getenv("MONGO_URI"),
        "mongodb://mongodb:27017/",
        "mongodb://localhost:27017/"
    ]
    for uri in uris:
        if not uri:
            continue
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
            print(f"MongoDB connecté avec succès via : {uri}")
            db = client["smart_home_energy"]
            return client, db["predictions"]
        except Exception:
            continue
    raise ConnectionError("Impossible de se connecter à MongoDB (testé mongodb:27017 et localhost:27017)")


mongo_client, predictions_collection = get_mongo_collection()

print("=" * 70)
print("MONGODB READY")
print("Database: smart_home_energy")
print("Collection: predictions")
print("=" * 70)



# ============================================================
# 2. Paths HDFS
# ============================================================

model_path = "hdfs://namenode:9000/smart-home/historical/models/random_forest_final"

historical_path = \
    "hdfs://namenode:9000/smart-home/historical/features/household_power_consumption_features"


# ============================================================
# 3. Load saved Random Forest
# ============================================================

model = RandomForestRegressionModel.load(model_path)

print("=" * 70)
print("RANDOM FOREST MODEL LOADED")
print("Number of trees:", model.getNumTrees)
print("Max depth:", model.getOrDefault(model.maxDepth))
print("=" * 70)


# ============================================================
# 4. Load last 1440 historical records (24 hours context)
# ============================================================

historical_df = spark.read.parquet(historical_path)

latest_history_df = historical_df \
    .orderBy(col("timestamp").desc()) \
    .limit(1440) \
    .orderBy(col("timestamp"))


# ============================================================
# 5. Create in-memory history deque
# ============================================================

history = deque(maxlen=1440)

historical_rows = latest_history_df.collect()

for row in historical_rows:
    history.append({
        "timestamp": row["timestamp"],
        "Global_active_power": float(row["Global_active_power"]),
        "Global_reactive_power": float(row["Global_reactive_power"]),
        "Voltage": float(row["Voltage"]),
        "Global_intensity": float(row["Global_intensity"]),
        "Sub_metering_1": float(row["Sub_metering_1"]),
        "Sub_metering_2": float(row["Sub_metering_2"]),
        "Sub_metering_3": float(row["Sub_metering_3"])
    })

print("HISTORICAL CONTEXT LOADED")
print("Historical rows in memory:", len(history))
print("=" * 70)


# ============================================================
# 6. Kafka JSON Schema étendu & tolérant
# ============================================================

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("Global_active_power", DoubleType(), True),
    StructField("Global_reactive_power", DoubleType(), True),
    StructField("Voltage", DoubleType(), True),
    StructField("Global_intensity", DoubleType(), True),
    StructField("Sub_metering_1", DoubleType(), True),
    StructField("Sub_metering_2", DoubleType(), True),
    StructField("Sub_metering_3", DoubleType(), True),
    # Alias supportés pour rétrocompatibilité
    StructField("power", DoubleType(), True),
    StructField("voltage", DoubleType(), True),
    StructField("current", DoubleType(), True)
])


# ============================================================
# 7. Read Kafka stream
# ============================================================

kafka_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092,localhost:9092")

kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_servers) \
    .option("subscribe", "smart-home-energy") \
    .option("startingOffsets", "latest") \
    .load()



# ============================================================
# 8. Parse JSON
# ============================================================

stream_df = kafka_df \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*")


# ============================================================
# Helper parsing functions
# ============================================================

def parse_timestamp(ts_value):
    """Parse de manière robuste les dates au format standard ou ISO."""
    if isinstance(ts_value, datetime):
        return ts_value
    if not ts_value:
        return datetime.now()
    ts_str = str(ts_value).strip()
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        pass
    # En cas de format hybride "YYYY-MM-DDTHH:MM:SS.microsec"
    clean_str = ts_str.replace("T", " ").split(".")[0]
    return datetime.strptime(clean_str, "%Y-%m-%d %H:%M:%S")


def extract_measurements(row):
    """Extrait les mesures avec mappage de secours automatique."""
    row_dict = row.asDict()

    # Active power (kW)
    power = row_dict.get("Global_active_power")
    if power is None:
        power = row_dict.get("power")
    power = float(power) if power is not None else 1.5

    # Voltage (V)
    voltage = row_dict.get("Voltage")
    if voltage is None:
        voltage = row_dict.get("voltage")
    voltage = float(voltage) if voltage is not None else 235.0

    # Intensity (A)
    intensity = row_dict.get("Global_intensity")
    if intensity is None:
        intensity = row_dict.get("current")
    if intensity is None:
        intensity = round((power * 1000.0) / voltage, 1)
    else:
        intensity = float(intensity)

    # Reactive power (kW)
    reactive = row_dict.get("Global_reactive_power")
    reactive = float(reactive) if reactive is not None else round(power * 0.1, 3)

    # Sub meterings (Wh)
    sub1 = float(row_dict["Sub_metering_1"]) if row_dict.get("Sub_metering_1") is not None else 0.0
    sub2 = float(row_dict["Sub_metering_2"]) if row_dict.get("Sub_metering_2") is not None else 0.0
    sub3 = float(row_dict["Sub_metering_3"]) if row_dict.get("Sub_metering_3") is not None else 0.0

    return power, reactive, voltage, intensity, sub1, sub2, sub3


# ============================================================
# 9. Feature + prediction function
# ============================================================

batch_number = 0


def process_batch(batch_df, batch_id):
    global batch_number

    batch_number += 1

    print()
    print("=" * 70)
    print(f"PROCESSING BATCH: {batch_number}")
    print("=" * 70)

    rows = batch_df.collect()

    if not rows:
        print("Batch vide.")
        return

    # Tri chronologique des messages
    rows = sorted(
        rows,
        key=lambda r: parse_timestamp(r["timestamp"])
    )

    predictions = []

    for row in rows:
        timestamp = parse_timestamp(row["timestamp"])

        (current_power, global_reactive, current_voltage,
         current_intensity, sub_1, sub_2, sub_3) = extract_measurements(row)

        # ----------------------------------------------------
        # Features retardées (Lags) depuis la deque history
        # ----------------------------------------------------
        lag_1 = history[-1]["Global_active_power"]
        lag_2 = history[-2]["Global_active_power"]
        lag_3 = history[-3]["Global_active_power"]
        lag_4 = history[-4]["Global_active_power"]
        lag_24 = history[-1440]["Global_active_power"]

        lag_voltage_1 = history[-1]["Voltage"]
        lag_intensity_1 = history[-1]["Global_intensity"]

        # ----------------------------------------------------
        # Rolling 3 hours (180 valeurs incluant la valeur courante)
        # ----------------------------------------------------
        values_3h = [
            x["Global_active_power"]
            for x in list(history)[-179:]
        ]
        values_3h.append(current_power)
        rolling_mean_3h = statistics.mean(values_3h)
        rolling_std_3h = statistics.stdev(values_3h) if len(values_3h) > 1 else 0.0

        # ----------------------------------------------------
        # Rolling 6 hours (360 valeurs incluant la valeur courante)
        # ----------------------------------------------------
        values_6h = [
            x["Global_active_power"]
            for x in list(history)[-359:]
        ]
        values_6h.append(current_power)
        rolling_mean_6h = statistics.mean(values_6h)

        # ----------------------------------------------------
        # Variables temporelles
        # Spark dayofweek: Sunday=1, Monday=2, ..., Saturday=7
        # ----------------------------------------------------
        python_weekday = timestamp.weekday()
        spark_day_of_week = ((python_weekday + 1) % 7) + 1
        hour = timestamp.hour
        day = timestamp.day
        month = timestamp.month
        is_weekend = 1 if spark_day_of_week in (1, 7) else 0

        # ----------------------------------------------------
        # Feature row
        # ----------------------------------------------------
        feature_data = [(
            timestamp,
            current_power,
            global_reactive,
            current_voltage,
            current_intensity,
            sub_1,
            sub_2,
            sub_3,
            lag_1,
            lag_2,
            lag_3,
            lag_4,
            lag_24,
            lag_voltage_1,
            lag_intensity_1,
            rolling_mean_3h,
            rolling_std_3h,
            rolling_mean_6h,
            hour,
            day,
            spark_day_of_week,
            month,
            is_weekend
        )]

        feature_columns = [
            "timestamp",
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

        feature_df = spark.createDataFrame(
            feature_data,
            feature_columns
        )

        # ----------------------------------------------------
        # Assemble les 22 features exactes du modèle
        # ----------------------------------------------------
        ml_features = [
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
            inputCols=ml_features,
            outputCol="features"
        )

        assembled_df = assembler.transform(feature_df)

        # ----------------------------------------------------
        # Prédiction MLlib
        # ----------------------------------------------------
        prediction_df = model.transform(assembled_df)

        result = prediction_df.select(
            "timestamp",
            "Global_active_power",
            "lag_1",
            "lag_24",
            "rolling_mean_3h",
            "rolling_mean_6h",
            "prediction"
        ).collect()[0]

        predictions.append(result)

        # ----------------------------------------------------
        # Mise à jour continue de l'historique en mémoire
        # ----------------------------------------------------
        history.append({
            "timestamp": timestamp,
            "Global_active_power": current_power,
            "Global_reactive_power": global_reactive,
            "Voltage": current_voltage,
            "Global_intensity": current_intensity,
            "Sub_metering_1": sub_1,
            "Sub_metering_2": sub_2,
            "Sub_metering_3": sub_3
        })

    # ========================================================
    # Affichage des prédictions
    # ========================================================
    print()
    print(
        "timestamp | current_power | lag_1 | lag_24 | "
        "rolling_3h | rolling_6h | prediction"
    )
    print("-" * 100)

    for result in predictions:
        print(
            str(result["timestamp"]),
            "|",
            result["Global_active_power"],
            "|",
            result["lag_1"],
            "|",
            result["lag_24"],
            "|",
            result["rolling_mean_3h"],
            "|",
            result["rolling_mean_6h"],
            "|",
            result["prediction"]
        )

    # ========================================================
    # Sauvegarde des prédictions dans MongoDB
    # ========================================================
    mongo_documents = []

    for result in predictions:
        mongo_documents.append({
            "timestamp": result["timestamp"],
            "current_power": float(result["Global_active_power"]),
            "lag_1": float(result["lag_1"]),
            "lag_24": float(result["lag_24"]),
            "rolling_mean_3h": float(result["rolling_mean_3h"]),
            "rolling_mean_6h": float(result["rolling_mean_6h"]),
            "prediction": float(result["prediction"]),
            "model": "random_forest_40_trees_depth_8"
        })

    if mongo_documents:
        predictions_collection.insert_many(mongo_documents)
        print()
        print(f"MONGODB: SAVED {len(mongo_documents)} PREDICTION(S) INTO COLLECTION 'predictions'")

    print()
    print("History size in memory:", len(history))
    print("=" * 70)


# ============================================================
# 10. Start streaming
# ============================================================

query = stream_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/smart-home-energy-checkpoint") \
    .start()

query.awaitTermination()

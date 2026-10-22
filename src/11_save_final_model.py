from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel

spark = SparkSession.builder \
    .appName("SaveFinalRandomForest") \
    .getOrCreate()

TRAIN_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/train"

MODEL_PATH = "hdfs://namenode:9000/smart-home/historical/models/random_forest_final"

print("=== Lecture du TRAIN ===")

train_df = spark.read.parquet(TRAIN_PATH)

print(f"Nombre de lignes TRAIN : {train_df.count()}")

print("=== Création du modèle final ===")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="target",
    predictionCol="prediction",
    numTrees=40,
    maxDepth=8,
    seed=42
)

print("=== Entraînement du modèle final ===")

model = rf.fit(train_df)

print("=== Modèle entraîné ===")

print("=== Sauvegarde du modèle dans HDFS ===")

model.write().overwrite().save(MODEL_PATH)

print(f"Modèle sauvegardé dans : {MODEL_PATH}")

print("=== Vérification ===")

loaded_model = RandomForestRegressionModel.load(MODEL_PATH)

print("Modèle rechargé avec succès.")
print(f"Nombre d'arbres : {loaded_model.getNumTrees}")
print(f"Profondeur maximale : {loaded_model.getOrDefault(loaded_model.maxDepth)}")

print("=== Sauvegarde terminée avec succès ===")

spark.stop()
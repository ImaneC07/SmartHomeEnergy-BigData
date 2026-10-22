from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel

spark = SparkSession.builder \
    .appName("PredictWithSavedModel") \
    .getOrCreate()

MODEL_PATH = "hdfs://namenode:9000/smart-home/historical/models/random_forest_final"
TEST_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/test"

# 1. Charger le modèle déjà entraîné
model = RandomForestRegressionModel.load(MODEL_PATH)

print("=== Modèle chargé ===")
print(f"Nombre d'arbres : {model.getNumTrees}")
print(f"Profondeur maximale : {model.getOrDefault(model.maxDepth)}")

# 2. Charger les données de test
test_df = spark.read.parquet(TEST_PATH)

# 3. Faire les prédictions
predictions = model.transform(test_df)

# 4. Afficher quelques résultats
print("=== Exemples de prédictions ===")

predictions.select(
    "timestamp",
    "target",
    "prediction"
).show(20, truncate=False)

spark.stop()
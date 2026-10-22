from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("RandomForestEnergyPrediction") \
    .getOrCreate()

TRAIN_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/train"
VAL_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/validation"
TEST_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/test"

print("=== Lecture des données ML ===")

train_df = spark.read.parquet(TRAIN_PATH)
val_df = spark.read.parquet(VAL_PATH)
test_df = spark.read.parquet(TEST_PATH)

print(f"TRAIN      : {train_df.count()}")
print(f"VALIDATION : {val_df.count()}")
print(f"TEST       : {test_df.count()}")

print("=== Création du Random Forest ===")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="target",
    predictionCol="prediction",
    numTrees=30,
    maxDepth=8,
    seed=42
)

print("=== Entraînement du modèle ===")

model = rf.fit(train_df)

print("=== Modèle entraîné avec succès ===")

print("=== Prédictions VALIDATION ===")

val_predictions = model.transform(val_df)

rmse_evaluator = RegressionEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="rmse"
)

mae_evaluator = RegressionEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="mae"
)

r2_evaluator = RegressionEvaluator(
    labelCol="target",
    predictionCol="prediction",
    metricName="r2"
)

val_rmse = rmse_evaluator.evaluate(val_predictions)
val_mae = mae_evaluator.evaluate(val_predictions)
val_r2 = r2_evaluator.evaluate(val_predictions)

print("=== Résultats VALIDATION ===")
print(f"Validation RMSE : {val_rmse}")
print(f"Validation MAE  : {val_mae}")
print(f"Validation R²   : {val_r2}")

print("=== Prédictions TEST ===")

test_predictions = model.transform(test_df)

test_rmse = rmse_evaluator.evaluate(test_predictions)
test_mae = mae_evaluator.evaluate(test_predictions)
test_r2 = r2_evaluator.evaluate(test_predictions)

print("=== Résultats TEST ===")
print(f"Test RMSE : {test_rmse}")
print(f"Test MAE  : {test_mae}")
print(f"Test R²   : {test_r2}")

print("=== Exemple de prédictions TEST ===")

test_predictions.select(
    "timestamp",
    "target",
    "prediction"
).show(10, truncate=False)

print("=== Random Forest terminé avec succès ===")

spark.stop()

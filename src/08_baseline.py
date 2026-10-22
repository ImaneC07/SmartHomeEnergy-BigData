from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("NaiveBaselineEnergyPrediction") \
    .getOrCreate()

TEST_PATH = "hdfs://namenode:9000/smart-home/historical/ml/test"

print("=== Lecture du TEST ===")

test_df = spark.read.parquet(TEST_PATH)

print(f"Nombre de lignes TEST : {test_df.count()}")

print("=== Création du baseline naïf ===")

baseline_df = test_df.withColumn(
    "prediction",
    test_df["lag_1"]
)

print("=== Exemple de prédictions ===")

baseline_df.select(
    "timestamp",
    "target",
    "lag_1",
    "prediction"
).show(10, truncate=False)

print("=== Évaluation du baseline ===")

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

rmse = rmse_evaluator.evaluate(baseline_df)
mae = mae_evaluator.evaluate(baseline_df)
r2 = r2_evaluator.evaluate(baseline_df)

print("=== Résultats Baseline ===")
print(f"Baseline RMSE : {rmse}")
print(f"Baseline MAE  : {mae}")
print(f"Baseline R²   : {r2}")

print("=== Baseline terminé avec succès ===")

spark.stop()
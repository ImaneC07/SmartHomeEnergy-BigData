from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("RandomForestHyperparameterTuning") \
    .getOrCreate()

TRAIN_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/train"
VAL_PATH = "hdfs://namenode:9000/smart-home/historical/ml_prepared/validation"

print("=== Lecture des données ===")

train_df = spark.read.parquet(TRAIN_PATH)
val_df = spark.read.parquet(VAL_PATH)

print(f"TRAIN      : {train_df.count()}")
print(f"VALIDATION : {val_df.count()}")

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

configurations = [
    (20, 6),
    (20, 8),
    (40, 6),
    (40, 8)
]

results = []

for num_trees, max_depth in configurations:

    print()
    print("========================================")
    print(f"TEST : numTrees={num_trees}, maxDepth={max_depth}")
    print("========================================")

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="target",
        predictionCol="prediction",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42
    )

    print("=== Entraînement ===")

    model = rf.fit(train_df)

    print("=== Évaluation VALIDATION ===")

    predictions = model.transform(val_df)

    rmse = rmse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    print(f"RMSE : {rmse}")
    print(f"MAE  : {mae}")
    print(f"R²   : {r2}")

    results.append(
        (num_trees, max_depth, rmse, mae, r2)
    )

print()
print("========================================")
print("RÉSULTATS DU HYPERPARAMETER TUNING")
print("========================================")

for num_trees, max_depth, rmse, mae, r2 in results:
    print(
        f"numTrees={num_trees}, "
        f"maxDepth={max_depth} | "
        f"RMSE={rmse:.6f} | "
        f"MAE={mae:.6f} | "
        f"R²={r2:.6f}"
    )

best_result = min(results, key=lambda x: x[2])

print()
print("========================================")
print("MEILLEURE CONFIGURATION")
print("========================================")

print(f"numTrees : {best_result[0]}")
print(f"maxDepth : {best_result[1]}")
print(f"RMSE     : {best_result[2]}")
print(f"MAE      : {best_result[3]}")
print(f"R²       : {best_result[4]}")

print()
print("=== Hyperparameter tuning terminé avec succès ===")

spark.stop()
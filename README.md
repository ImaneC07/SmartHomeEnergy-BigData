# SmartHomeEnergy ⚡
### Real-Time Household Electricity Consumption Forecasting with Big Data

SmartHomeEnergy is an academic Big Data project that combines historical electricity consumption data, distributed processing, machine learning, and simulated real-time data streams to forecast household power consumption.

The project demonstrates a Lambda-style architecture using Apache Hadoop, Apache Spark, Apache Kafka, MongoDB, and Streamlit.

## Project Overview

Household electricity consumption varies throughout the day according to user activity and electrical equipment usage. Analyzing historical measurements and processing incoming data can help explore consumption patterns and demonstrate predictive monitoring.

**Project objectives**
- Process a large historical electricity consumption dataset.
- Clean data and engineer temporal and lag-based features.
- Train and evaluate machine learning models for power forecasting.
- Simulate an IoT data stream using Apache Kafka.
- Apply a saved Spark MLlib model to streaming data.
- Store predictions in MongoDB and visualize them in a Streamlit dashboard.

## Architecture

### Historical data pipeline

```text
UCI Household Power Consumption Dataset
                  |
                  v
          HDFS (raw data)
                  |
                  v
      Apache Spark processing
       Cleaning and features
                  |
                  v
       Train / validation / test
                  |
                  v
       Model training and evaluation
                  |
                  v
          Saved model in HDFS
```

### Real-time prediction pipeline

```text
Simulated IoT Producer
          |
          v
   Apache Kafka
          |
          v
Spark Structured Streaming
          |
          v
  Random Forest (MLlib)
          |
          v
       MongoDB
          |
          v
   Streamlit Dashboard
```

The historical and streaming workflows share the forecasting model. The streaming application loads the trained model from HDFS and uses historical records to initialize its rolling feature history.

## Technology Stack

| Component | Technology |
|---|---|
| Distributed storage | Apache Hadoop HDFS 3.3.6 |
| Stream ingestion | Apache Kafka 4.0.1 |
| Distributed processing | Apache Spark 3.5.1 |
| Machine learning | Spark MLlib — Random Forest Regressor |
| Prediction storage | MongoDB 7 |
| Dashboard | Streamlit |
| Programming language | Python |
| Environment | Docker Compose |

## Dataset

The project uses the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- Collection period: December 2006 to November 2010
- Original dataset size: approximately 2.07 million measurements
- Raw file: `household_power_consumption.txt`

The raw dataset is not included in this repository because of its size. Follow the instructions in [`data/raw/README.md`](data/raw/README.md) to obtain it.

## Feature Engineering

The forecasting workflow uses several categories of features:

- **Electrical measurements:** active and reactive power, voltage, current intensity, and sub-metering values.
- **Lag features:** previous measurements used to capture recent consumption patterns.
- **Rolling statistics:** rolling means and standard deviations.
- **Calendar features:** hour, day, day of week, month, and weekend indicator.

The final feature set and its exact ordering must remain consistent between model training and streaming inference.

## Model and Reported Results

The project uses a Random Forest Regressor implemented with Spark MLlib.

| Setting | Value |
|---|---|
| Number of trees | 40 |
| Maximum tree depth | 8 |
| Random seed | 42 |
| Input features | 22, as configured in the project |

The following results were reported in the project experiments:

| Metric | Reported result |
|---|---:|
| R² | 0.9376 |
| RMSE | 0.2152 kW |
| MAE | 0.0828 kW |
| Prediction latency | 463.9 ms |

These figures describe the reported experimental results, not a guarantee of performance on other datasets or environments. Reproduce the evaluation pipeline to independently verify them. Latency should be interpreted according to the measurement procedure used in the experiment.

## Repository Structure

```text
.
├── dashboard.py
├── docker-compose.yml
├── requirements.txt
├── run_streaming.bat
├── run_streaming.ps1
├── data/
│   └── raw/
│       └── README.md
├── hadoop/
│   └── config/
│       ├── core-site.xml
│       └── hdfs-site.xml
├── model_export/
├── spark/
│   ├── Dockerfile
│   └── jars/
└── src/
    ├── 01_clean_historical.py
    ├── 02_quality_check.py
    ├── 03_feature_engineering.py
    ├── 04_feature_quality_check.py
    ├── 05_train_val_test_split.py
    ├── 06_prepare_ml_data.py
    ├── 07_train_linear_regression.py
    ├── 08_baseline.py
    ├── 08_train_random_forest.py
    ├── 09_tune_random_forest.py
    ├── 10_train_final_model.py
    ├── 11_save_final_model.py
    ├── 12_predict_with_saved_model.py
    ├── 13_streaming_real_features.py
    └── iot_producer.py
```

The `src/` directory also contains supporting test and utility scripts. The local `model_export/` directory is an auxiliary export; the real-time application loads its model from HDFS.

## Prerequisites

- Windows, Linux, or macOS with Docker Desktop or Docker Engine and Docker Compose.
- Python 3.11 recommended for the host-side producer and dashboard.
- Git.
- Sufficient RAM and disk space for Hadoop, Spark, Kafka, MongoDB, and the dataset. Resource requirements depend on the environment.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with the URL and directory name of your published repository.

### 2. Download the dataset

Follow [`data/raw/README.md`](data/raw/README.md) and place the extracted file at:

```text
data/raw/household_power_consumption.txt
```

### 3. Start the infrastructure

Build the custom Spark image and start the services:

```bash
docker compose up -d --build
```

Check container status:

```bash
docker compose ps
```

Useful local endpoints:

- HDFS NameNode UI: http://localhost:9870
- Spark UI: http://localhost:8080
- Kafka bootstrap address for host applications: `localhost:9092`
- MongoDB: `localhost:27017`

The services starting successfully does not mean that historical data or a trained model is already available in HDFS.

### 4. Prepare historical data and train the model

The numbered scripts in `src/` implement the historical processing, data quality checks, feature engineering, dataset splitting, model training, and evaluation workflow.

Run the relevant scripts with `spark-submit` in the intended order, ensuring that each stage's HDFS output exists before starting the next stage. The final model must be saved to:

```text
hdfs://namenode:9000/smart-home/historical/models/random_forest_final
```

The streaming application also expects historical features at:

```text
hdfs://namenode:9000/smart-home/historical/features/household_power_consumption_features
```

The precise commands depend on the available input files and the HDFS paths created by the preceding stages. Check each script's input and output paths before running the workflow on a fresh installation.

### 5. Install host-side Python dependencies

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 6. Start real-time prediction

Make sure Kafka is running, the model and historical features exist in HDFS, and the Spark container is ready.

On Windows, run:

```powershell
.\run_streaming.bat
```

The script submits `13_streaming_real_features.py` inside the Spark container.

### 7. Start the simulated IoT producer

In a separate terminal, with the virtual environment activated:

```bash
python src/iot_producer.py
```

The producer sends simulated electrical measurements to the Kafka topic `smart-home-energy` every two seconds. It connects to the host address `localhost:9092`.

### 8. Start the dashboard

In another terminal:

```bash
streamlit run dashboard.py
```

Open the local Streamlit URL displayed in the terminal, usually http://localhost:8501.

The dashboard displays data stored in MongoDB. It may remain empty until the producer and streaming application are running successfully.

## Important Notes

- The IoT producer generates synthetic measurements; it does not collect data from physical sensors.
- The streaming workflow depends on historical data and a model already available in HDFS.
- The Docker Compose configuration is intended for local development and demonstration, not production deployment.
- The repository does not include the large raw dataset.


*Built as a hands-on exploration of distributed data processing, machine learning, and real-time analytics.*

@echo off
echo ============================================================
echo   Smart Home Energy - Demarrage Spark Streaming (Docker)
echo ============================================================
echo Execution dans le conteneur smarthome-spark...
echo Appuyez sur Ctrl+C pour arreter le traitement.
echo ============================================================
docker exec -it smarthome-spark /opt/spark/bin/spark-submit /tmp/spark-apps/src/13_streaming_real_features.py

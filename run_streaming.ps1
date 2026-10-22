Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Smart Home Energy - Demarrage Spark Streaming (Docker)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Execution dans le conteneur smarthome-spark..." -ForegroundColor Green
Write-Host "Appuyez sur Ctrl+C pour arreter le traitement." -ForegroundColor DarkGray

docker exec -it smarthome-spark /opt/spark/bin/spark-submit /tmp/spark-apps/src/13_streaming_real_features.py

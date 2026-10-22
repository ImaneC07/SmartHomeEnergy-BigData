import json
from kafka import KafkaConsumer

TOPIC = "smart-home-energy"
OUTPUT_FILE = "data/kafka_raw.json"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="hdfs-consumer"
)

print("Kafka → HDFS Consumer démarré...")
print(f"Les données seront sauvegardées dans : {OUTPUT_FILE}")

with open(OUTPUT_FILE, "a", encoding="utf-8") as file:

    for message in consumer:
        data = message.value

        file.write(json.dumps(data) + "\n")
        file.flush()

        print(f"Message sauvegardé : {data}")
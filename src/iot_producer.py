import json
import random
import time
from datetime import datetime

from kafka import KafkaProducer


# Connexion au broker Kafka (support localhost hôte ou conteneur kafka)
bootstrap_servers = "localhost:9092"
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    value_serializer=lambda value: json.dumps(value).encode("utf-8")
)

TOPIC = "smart-home-energy"


print("=" * 60)
print("IoT Smart Home Producer démarré...")
print(f"Broker Kafka : {bootstrap_servers}")
print(f"Topic cible  : {TOPIC}")
print("Envoi des télémesures électriques en temps réel...")
print("Appuyez sur Ctrl+C pour arrêter.")
print("=" * 60)


try:
    while True:

        # --------------------------------------------------------
        # Simulation réaliste des équipements d'une maison intelligente
        # --------------------------------------------------------
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Sous-compteurs d'énergie (Wh)
        sub_1 = round(random.choice([0.0, 0.0, 0.0, random.uniform(1.0, 4.0)]), 1)   # Cuisine
        sub_2 = round(random.choice([0.0, 0.0, 1.0, random.uniform(1.0, 2.5)]), 1)   # Buanderie
        sub_3 = round(random.choice([0.0, random.uniform(10.0, 18.0)]), 1)            # Chauffe-eau / Climatisation

        # Puissance active totale (kW) liée aux sous-compteurs + bruit de fond
        base_power = round(random.uniform(0.3, 0.8), 3)
        sub_power_kw = (sub_1 + sub_2 + sub_3) * 60 / 1000.0
        global_active_power = round(base_power + sub_power_kw, 3)

        # Tension secteur (Volts)
        voltage = round(random.uniform(232.0, 242.0), 2)

        # Puissance réactive (kW)
        global_reactive_power = round(random.uniform(0.06, 0.25), 3)

        # Intensité de courant globale (Ampères) : I = P / V
        global_intensity = round((global_active_power * 1000.0) / voltage, 1)

        # Payload complet conforme au modèle Spark MLlib & rétrocompatible
        message = {
            "timestamp": timestamp_str,
            "Global_active_power": global_active_power,
            "Global_reactive_power": global_reactive_power,
            "Voltage": voltage,
            "Global_intensity": global_intensity,
            "Sub_metering_1": sub_1,
            "Sub_metering_2": sub_2,
            "Sub_metering_3": sub_3,
            # Clés alias pour compatibilité
            "power": global_active_power,
            "voltage": voltage,
            "current": global_intensity
        }

        # Envoi vers Kafka
        producer.send(TOPIC, value=message)
        producer.flush()

        print(
            f"[{timestamp_str}] Envoyé -> P: {global_active_power} kW | "
            f"V: {voltage} V | I: {global_intensity} A | "
            f"Subs: ({sub_1}, {sub_2}, {sub_3})"
        )

        time.sleep(2)

except KeyboardInterrupt:
    print("\nIoT Producer arrêté.")

finally:
    producer.close()
 
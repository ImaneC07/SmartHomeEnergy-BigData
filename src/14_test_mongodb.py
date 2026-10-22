import os
import socket
from pymongo import MongoClient

def get_mongo_client():
    uris = [
        os.getenv("MONGO_URI"),
        "mongodb://localhost:27017/",
        "mongodb://mongodb:27017/"
    ]
    for uri in uris:
        if not uri:
            continue
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            return client, uri
        except Exception:
            continue
    raise ConnectionError("Impossible de se connecter à MongoDB (testé localhost et mongodb:27017)")

client, connected_uri = get_mongo_client()

db = client["smart_home_energy"]
collection = db["test"]

document = {
    "message": "MongoDB connection works!",
    "source": f"Tested from {connected_uri}"
}

result = collection.insert_one(document)

print("=" * 60)
print("MONGODB CONNECTION SUCCESSFUL")
print("=" * 60)
print("Connected URI:", connected_uri)
print("Inserted document ID:", result.inserted_id)
print("Database:", db.name)
print("Collection:", collection.name)
print("=" * 60)

client.close()
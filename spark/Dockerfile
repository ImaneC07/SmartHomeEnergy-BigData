FROM apache/spark:3.5.1

USER root

RUN pip install --no-cache-dir numpy pymongo

COPY jars/*.jar /opt/spark/jars/

RUN chmod 644 /opt/spark/jars/*.jar

USER spark

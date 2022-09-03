FROM python:3.7.0

RUN pip install mlflow==1.8.0 psycopg2==2.8.* boto3==1.13.0

RUN mkdir -p /server
WORKDIR /server
COPY ./wait-for-it.sh /server

RUN chmod +x wait-for-it.sh

RUN mkdir /mlflow/

EXPOSE 5000

CMD ./wait-for-it.sh postgres:5432 -- mlflow server \
# --backend-store-uri /mlflow \
    --backend-store-uri postgresql://ivnard:Orion123@postgres:5432/mlflow \
    # --default-artifact-root /mlflow \
    --default-artifact-root s3://mlflow/artifacts \
    --host 0.0.0.0
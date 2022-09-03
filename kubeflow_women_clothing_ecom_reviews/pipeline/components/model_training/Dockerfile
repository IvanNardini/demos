FROM tensorflow/tensorflow:2.4.1
WORKDIR usr/local/pipeline/components/model_trainer
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY component .
EXPOSE 9090
ENV PYTHONPATH "${PYTHONPATH}:usr/local/pipeline/components/collect/src"
ENTRYPOINT [ "python3", "run_train.py" ]
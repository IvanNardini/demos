FROM tensorflow/tensorflow:2.4.1
WORKDIR usr/local/pipeline/components/prepare
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY component .
ENV PYTHONPATH "${PYTHONPATH}:usr/local/pipeline/components/prepare/src"
ENTRYPOINT [ "python", "./run_prepare.py" ]
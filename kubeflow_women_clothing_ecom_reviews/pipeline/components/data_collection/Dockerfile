FROM tensorflow/tensorflow:2.4.1
WORKDIR usr/local/pipeline/components/collect
RUN pip install --upgrade pip
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY kaggle.json /root/.kaggle/
RUN chmod 600 /root/.kaggle/kaggle.json
COPY component .
EXPOSE 9090
ENV PYTHONPATH "${PYTHONPATH}:usr/local/pipeline/components/collect/src"
ENTRYPOINT [ "python3", "run_collect.py" ]
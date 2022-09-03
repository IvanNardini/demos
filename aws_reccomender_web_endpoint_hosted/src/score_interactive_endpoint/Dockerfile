
FROM python:3.8
    
WORKDIR /app

ADD . /app

RUN pip install -r ./requirements.txt

EXPOSE 8050

ENTRYPOINT [ "python" ]

CMD [ "run.py" ]

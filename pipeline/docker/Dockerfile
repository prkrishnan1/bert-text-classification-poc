FROM python:3.9.7-slim-bullseye

WORKDIR /opt/ml/processing/

COPY ./requirements.txt .

RUN pip3 install -r requirements.txt
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["/usr/local/bin/python3"]

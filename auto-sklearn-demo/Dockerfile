FROM python:3-slim
LABEL maintainer="liutony@seas.upenn.edu"

RUN apt-get update; apt-get install -y curl
RUN apt-get install -y build-essential swig gcc
RUN curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install

RUN pip install auto-sklearn

COPY automl.py .

CMD python -W ignore automl.py
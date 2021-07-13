FROM python:3.8.6-buster

# FROM nvidia/cuda:10.2-devel-ubuntu18.04

#RUN apt-get update && \
#	apt-get install -y curl python3.8 python3.8-distutils && \
#	ln -s /usr/bin/python3.8 /usr/bin/python && \
#	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install -r requirements.txt

RUN apt-get update && apt-get -qq -y install xz-utils wget vim
RUN wget -q https://www.dropbox.com/s/4lbyhjlotrz7nn6/text_transformer.tar.xz?dl=1 \
    -O $PROJECT_ROOT/lib/models/text_transformer.tar.xz
RUN tar xf $PROJECT_ROOT/lib/models/text_transformer.tar.xz \
    -C $PROJECT_ROOT/lib/models/

CMD python lib/run.py

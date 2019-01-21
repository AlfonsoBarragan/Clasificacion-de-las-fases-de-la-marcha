FROM python:3.6-slim-stretch
LABEL maintainer="Alf"

ADD setup/requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
        git \
        locales \
        libgdal20 \ 
        libgdal-dev \ 
        libsm6 \
        libxext6 \
        libxrender-dev && \
    python -m pip install numpy cython --no-binary numpy,cython && \
    pip install -r /tmp/requirements.txt && \
    mkdir /root/work

CMD ["/bin/bash"]


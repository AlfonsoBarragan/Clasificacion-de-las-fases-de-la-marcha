FROM debian:latest
LABEL maintainer="Alf"

COPY setup /home/

RUN apt-get update && mkdir /home/work/ && sh /home/install-requirements.sh && pip3 install -r /home/requirements.txt && rm /home/install-requirements.sh /home/requirements.txt

CMD /bin/bash

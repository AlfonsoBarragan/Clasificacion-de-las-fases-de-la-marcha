FROM debian:latest
LABEL maintainer="Alf"

COPY setup /home/

RUN apt-get update && mkdir /home/work/ && sh /home/install-requirements.sh

CMD /bin/bash

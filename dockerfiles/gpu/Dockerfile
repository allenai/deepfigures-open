# docker image for running DeepFigures on a gpu

FROM tensorflow/tensorflow:latest-gpu-py3

ENV LC_ALL C.UTF-8

# install system packages

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN apt-get install -y libjasper1 libjasper-dev

RUN apt-get clean \
 && apt-get update --fix-missing \
 && apt-get install -y \
    git \
    curl \
    gcc \
    build-essential \
    tcl \
    g++ \
    zlib1g-dev \
    libjpeg8-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng12-dev \
    tcl-dev \
    tk-dev \
    python3 \
    python3-pip \
    python3-tk \
    ghostscript \
    openjdk-8-jre \
    poppler-utils \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    texlive-font-utils

WORKDIR /work

# install python packages

ADD ./requirements.txt /work

RUN pip3 install --upgrade pip \
 && pip install Cython==0.25.2
RUN pip3 install -r ./requirements.txt

ADD ./vendor/tensorboxresnet /work/vendor/tensorboxresnet
RUN pip3 install -e /work/vendor/tensorboxresnet

ADD . /work
RUN pip3 install --quiet -e /work

RUN pip3 install Pillow

RUN apt-get install curl
RUN curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN mkdir ~/.aws
ADD credentials /work
RUN cat /work/credentials > ~/.aws/credentials

CMD [ "/bin/bash" ]

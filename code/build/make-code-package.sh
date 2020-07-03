#!/usr/bin/env bash

CODE=code

VERSION_NUMBER=0.0.1
TIMESTAMP=$(date +%y%m%d%H%M%S)
CONTAINER="build-deb-"$(echo $RANDOM % 100 + 1 | bc)

CURRENTUSER=$(id -u $USERNAME)

PACKAGE_PATH="/opt/${CODE}"

docker build --tag $CODE \
             --file Dockerfile.code .

docker run -t --name $CONTAINER \
    --volume $(pwd)/../python:/python \
    --volume $(pwd):/build \
    $CODE \
    bash -c "python3 -m venv ${PACKAGE_PATH}/venv \
    && ${PACKAGE_PATH}/venv/bin/pip3 install --upgrade pip \
    && ${PACKAGE_PATH}/venv/bin/pip3 install wheel \
    && ${PACKAGE_PATH}/venv/bin/pip3 install --index-url https://******/repository/pypi-all/simple \
                                        -r /python/requirements.txt\
    && ${PACKAGE_PATH}/venv/bin/pip3 install --no-deps --index-url https://******/repository/pypi-all/simple \
                                        -r /source/requirements-ml.txt\
    && ${PACKAGE_PATH}/venv/bin/pip3 install --index-url https://******/repository/pypi-all/simple \
                                        -r /source/requirements-extra.txt\
    && fpm \
              -s dir \
              -t deb \
              --deb-user yexp \
              --deb-group yexp \
              -n ${CODE} \
              -v ${VERSION_NUMBER} \
              --iteration ${TIMESTAMP} \
              --description 'CODE' \
              -p /build \
              ${PACKAGE_PATH}/=${PACKAGE_PATH} \
              /python/=${PACKAGE_PATH}/python/ \
              && chown ${CURRENTUSER} /build/*.deb"

docker rm -f $CONTAINER

exit 0

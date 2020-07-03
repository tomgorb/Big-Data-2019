#!/usr/bin/env bash

DAG=dag

VERSION_NUMBER=0.0.1
TIMESTAMP=$(date +%y%m%d%H%M%S)
CONTAINER="build-deb-"$(echo $RANDOM % 100 + 1 | bc)

CURRENTUSER=$(id -u $USERNAME)

PACKAGE_PATH="/opt/airflow/dags/${DAG}"

docker build --tag $DAG \
             --file Dockerfile.dag .

docker run -t --name $CONTAINER \
    --volume $(pwd)/../dags:/dags \
    --volume $(pwd):/build \
    $DAG \
    bash -c "fpm \
              -s dir \
              -t deb \
              --deb-user yexp \
              --deb-group yexp \
              -n ${DAG} \
              -v ${VERSION_NUMBER} \
              --iteration ${TIMESTAMP} \
              --description 'DAG' \
              -p /build \
              /dags/${DAG}.py=${PACKAGE_PATH}/${DAG}.py\
              && chown ${CURRENTUSER} /build/*.deb"

docker rm -f $CONTAINER

exit 0
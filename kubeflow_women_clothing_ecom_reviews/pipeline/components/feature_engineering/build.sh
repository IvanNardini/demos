#!/usr/bin/env bash

DOCKER_IMAGE=${1:-feature_generator:latest}
DOCKER_HOST=${2:-docker.io}
DOCKER_REPO=${3:-in92}
MODE=${4}

docker build -t "$DOCKER_IMAGE" .
docker tag "$DOCKER_IMAGE" "$DOCKER_HOST/$DOCKER_REPO/$DOCKER_IMAGE"
#docker push "$DOCKER_HOST/$DOCKER_IMAGE"
#
#if [ -z "$MODE" ]; then
#  docker container run --rm -it -p 9090:9090 -v $(pwd):/usr/src/app $DOCKER_IMAGE --config config.yaml --mode local
#fi
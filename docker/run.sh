#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=${DIR%/docker}

nvidia-docker run \
--ipc=host \
-v $DIR:/root/inequity-release \
-w /root/inequity-release \
-it benjaminrwilson/inequity-release:latest

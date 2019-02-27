#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=${DIR%/scripts}

mkdir $DIR/datasets && \
wget -O $DIR/annotations.zip https://s3.amazonaws.com/inequity-release/annotations.zip && \
wget -O $DIR/bdd100k.zip https://s3.amazonaws.com/inequity-release/bdd100k.zip && \
wget -O $DIR/weights.zip https://s3.amazonaws.com/inequity-release/weights.zip && \
unzip $DIR/annotations.zip -d $DIR/datasets && \
unzip $DIR/bdd100k.zip -d $DIR/datasets && \
unzip $DIR/weights.zip -d $DIR/ && \
rm $DIR/annotations.zip && \
rm $DIR/bdd100k.zip
rm $DIR/weights.zip

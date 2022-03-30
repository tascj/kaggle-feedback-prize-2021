#!/usr/bin/env bash

docker build docker/ -f docker/Dockerfile --rm \
    -t $USER/feedback-prize-2021

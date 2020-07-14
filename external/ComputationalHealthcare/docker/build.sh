#!/usr/bin/env bash
set -x
docker rm $(docker ps -qa --no-trunc --filter "status=exited")
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
docker build -t computationalhealthcare:latest .

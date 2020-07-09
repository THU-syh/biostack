#!/usr/bin/env bash
set -xe
docker cp NRD_2013_Core.CSV computational-healthcare:/root/data/CH/NRD/RAW/
docker exec -u="root" -it computational-healthcare fab prepare_nrd
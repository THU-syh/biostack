#!/usr/bin/env bash
set -xe
docker cp PUDF_base1q2006_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base1q2007_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base1q2008_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base1q2009_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base2q2006_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base2q2007_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base2q2008_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base2q2009_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base3q2006_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base3q2007_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base3q2008_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base3q2009_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base4q2006_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base4q2007_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base4q2008_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker cp PUDF_base4q2009_tab.txt.gz computational-healthcare:/root/data/CH/TX/RAW/
docker exec -u="root" -it computational-healthcare fab prepare_tx
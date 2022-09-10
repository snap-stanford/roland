#!/bin/bash
# This script downloads the Roland public data set from the online resources.

# create a directory for the data.
mkdir ./roland_public_data
cd ./roland_public_data

# download bitcoin alpha data.
wget "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz"
# the gzip -d command unzip .gz file on mac, you might need to use another command to unzip gz files on your specific system.
gzip -d soc-sign-bitcoinalpha.csv.gz
mv soc-sign-bitcoinalpha.csv bitcoinalpha.csv

# download bitcoin OTC data.
wget "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
gzip -d soc-sign-bitcoinotc.csv.gz
mv soc-sign-bitcoinotc.csv bitcoinotc.csv

# download college message data.
wget "http://snap.stanford.edu/data/CollegeMsg.txt.gz"
gzip -d CollegeMsg.txt.gz

# download reddit data.
wget "http://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
mv soc-redditHyperlinks-body.tsv reddit-body.tsv
wget "http://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv"
mv soc-redditHyperlinks-title.tsv reddit-title.tsv
wget "http://snap.stanford.edu/data/web-redditEmbeddings-subreddits.csv"

# verify the MD5 of datasets.
# use md5sum command on ubuntu.
for file in `ls`;
    do md5sum $file;
done;

# use md5 command on Mac OS.
for file in `ls`;
    do md5 $file;
done;

# expected MD5 values:
# 7254104d8e8246aae82408c8984d8267  bitcoinalpha.csv
# eeaf5cd1d29ab435505baeeb6816317b  bitcoinotc.csv
# cac6751a872aa91d1d1fe24cede118dd  CollegeMsg.txt
# 4ed2bb05718498b66450a10216e29b86  reddit-body.tsv
# e1babc01eb264be47cbab4fe134546de  reddit-title.tsv
# f9abaa5a3a238920a208f1bc8b62672e  web-redditEmbeddings-subreddits.csv

# You can download the bigger AS733 dataset as well.
wget "http://snap.stanford.edu/data/as-733.tar.gz"
# decompress the file
tar -xzvf ./as-733.tar.gz

# verify the MD5 of datasets.
# There are 733 files in total, please refer to the public_dataset_MD5.txt for MD5 values of each file.
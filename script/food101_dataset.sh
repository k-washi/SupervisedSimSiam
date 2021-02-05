#!/bin/sh

mkdir dataset
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz -O ./dataset/dataset.tar.gz
tar -zxvf ./dataset/dataset.tar.gz -C ./dataset/


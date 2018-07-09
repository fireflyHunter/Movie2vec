#!/bin/bash
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
mkdir ml-20m
mkdir ml-10m
unzip ml-20m.zip -d ./ml-20m
unzip ml-10m.zip -d ./ml-10m
python3 preprocess.py ./ml-20m/ml-20m/ratings.csv ml-20m/ml-20m/ratings_data.csv ml-20m/ml-20m/movie_sents.txt
python3 preprocess.py ./ml-10m/ml-10M100K/ratings.dat ml-10m/ml-10M100K/ratings_data.csv ml-10m/ml-10M100K/movie_sents.txt
rm ml-20m.zip
rm ml-10m.zip
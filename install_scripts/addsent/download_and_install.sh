#!/bin/bash
pip install -r ./install_scripts/addsent/requirements.txt

cd ./data
### GloVe vectors ###
if [ ! -d glove ]
then
  mkdir glove
  cd glove
  wget http://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip
  cd ..
fi
### SQuAD ###
if [ ! -d squad ]
then
  mkdir squad
  cd squad
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  cd ..
fi
cd ..

mkdir -p lib
cd lib
# CoreNLP 3.9.1
corenlp='stanford-corenlp-full-2018-02-27' #'stanford-corenlp-full-2015-12-09'
if [ ! -d "${corenlp}" ]
then
  wget "http://nlp.stanford.edu/software/${corenlp}.zip"
  unzip "${corenlp}.zip"
  ln -s "${corenlp}" stanford-corenlp
fi
cd ..

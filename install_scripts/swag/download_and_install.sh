#!/bin/bash
pip install -r ./install_scripts/swag/requirements.txt

cd ./data
### ActivityNet Captions ###
if [ ! -d activitynet ]
then
  mkdir activitynet
  cd activitynet
  wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
  unzip captions.zip
  cd ..
fi
cd ..

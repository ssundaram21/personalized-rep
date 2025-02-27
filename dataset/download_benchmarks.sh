# Downloads and unzips three evaluation datasets for personalized representations

#!/bin/bash
mkdir -p ./data
cd data

# Download pods and dogs datasets
echo "Downloading zips"
wget -O pods.zip https://data.csail.mit.edu/personal_rep/pods.zip
wget -O dogs.zip https://data.csail.mit.edu/personal_rep/dogs.zip

# Unzip
echo "Unzipping data"
unzip pods.zip
unzip dogs.zip
rm pods.zip
rm dogs.zip

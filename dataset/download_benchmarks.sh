# Downloads and unzips three evaluation datasets for personalized representations

#!/bin/bash
mkdir -p ./data
cd data

# Download datasets
echo "Downloading zips"
wget -O pods.zip https://data.csail.mit.edu/personal_rep/pods.zip
wget -O dogs.zip https://data.csail.mit.edu/personal_rep/dogs.zip
wget -O df2.zip https://data.csail.mit.edu/personal_rep/df2.zip

# Unzip
echo "Unzipping data"
unzip pods.zip
unzip dogs.zip
unzip df2.zip
rm pods.zip
rm dogs.zip
rm df2.zip
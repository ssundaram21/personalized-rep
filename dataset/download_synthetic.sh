# Downloads and unzips the best-performing synthetic datasets for DF2, Dogs, and PODS

#!/bin/bash
mkdir -p ./synthetic_data
cd synthetic_data

# Download datasets
echo "Downloading zips"
wget -O pods_dreambooth_llm_masked_filtered_cfg_5.zip https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_dreambooth_llm_masked_filtered_cfg_5.zip
wget -O pods_negatives.zip https://data.csail.mit.edu/personal_rep/syn_data/pods/pods_negatives.zip
wget -O dogs_dreambooth_llm_masked_filtered_cfg_5.zip https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_dreambooth_llm_masked_filtered_cfg_5.zip
wget -O dogs_negatives.zip https://data.csail.mit.edu/personal_rep/syn_data/dogs/dogs_negatives.zip
wget -O df2_dreambooth_llm_masked_filtered_cfg_5.zip https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_dreambooth_llm_masked_filtered_cfg_5.zip
wget -O df2_negatives.zip https://data.csail.mit.edu/personal_rep/syn_data/df2/df2_negatives.zip

# Unzip
echo "Unzipping data"
unzip pods_dreambooth_llm_masked_filtered_cfg_5.zip
unzip pods_negatives.zip
unzip dogs_dreambooth_llm_masked_filtered_cfg_5.zip
unzip dogs_negatives.zip
unzip df2_dreambooth_llm_masked_filtered_cfg_5.zip
unzip df2_negatives.zip

rm pods_dreambooth_llm_masked_filtered_cfg_5.zip
rm pods_negatives.zip
rm dogs_dreambooth_llm_masked_filtered_cfg_5.zip
rm dogs_negatives.zip
rm df2_dreambooth_llm_masked_filtered_cfg_5.zip
rm df2_negatives.zip
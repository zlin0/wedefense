#!/bin/bash
# set -x
ASVspoof2019_dir=$1
RED='\033[0;31m'
link=https://datashare.ed.ac.uk/download/DS_10283_3336.zip

if [ ! -d ${ASVspoof2019_dir} ]; then
    mkdir -p "${ASVspoof2019_dir}"

    cd ${ASVspoof2019_dir}
    wget -q --show-progress -c "${link}" -O asvspoof2019.zip
    unzip ${ASVspoof2019_dir}/asvspoof2019.zip
    unzip LA.zip
    # unzip PA.zip, We currently only focus on asvspoof2019

    cd -

fi


echo -e "${RED} We have ASVspoof2019 database on ${ASVspoof2019_dir} now"

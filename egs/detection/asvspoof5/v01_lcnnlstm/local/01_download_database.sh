#!/bin/bash
# set -x
ASVspoof5_dir=$1
RED='\033[0;31m'
echo "${ASVspoof5_dir}"

for dset in T D E; do
    if [ "$dset" = "E" ]; then
        FILE_NAMEs="a b c d e f g h i j"
    elif [ "$dset" = "D" ]; then
        FILE_NAMEs="a b c"
    elif [ "$dset" = "T" ]; then
        FILE_NAMEs="a b c d e"
    fi

    for name in ${FILE_NAMEs}; do    
        fname=flac_${dset}_a${name}.tar
        link="https://zenodo.org/records/14498691/files/${fname}?download=1"
        echo -e "${RED}Downloading ${fname}"
        echo ${link}
        wget -q --show-progress -c "${link}" -O "${ASVspoof5_dir}/${fname}"
        tar -xf "${ASVspoof5_dir}/${fname}" -C "${ASVspoof5_dir}"
        rm "${ASVspoof5_dir}/${fname}"
    done
done

fname=ASVspoof5_protocols.tar.gz
link="https://zenodo.org/records/14498691/files/${fname}?download=1"
echo -e "${RED}Downloading ${fname}"
wget -q --show-progress -c ${link} -O "${ASVspoof5_dir}/${fname}"
tar -xf "${ASVspoof5_dir}/${fname}" -C "${ASVspoof5_dir}"
rm "${ASVspoof5_dir}/${fname}"

OTH_files="LICENSE.txt README.txt"
for fname in $OTH_files; do
    link="https://zenodo.org/records/14498691/files/${fname}?download=1"
    wget -q --show-progress -c "${link}" -O "${ASVspoof5_dir}/${fname}"
    echo -e "${RED}Downloading ${fname}"
done	


echo -e "${RED} We have ASVspoof5 database on ${ASVspoof5_dir} now"

#!/bin/bash
set -x
ASVspoof5_dir=$1
dset=$2

if [ "$dset" = "E"  ]; then
    FILE_NAMEs="a b c d e f g h i j"
elif [ "$dset" = "D" ]; then
    FILE_NAMEs="a b c"
else
    FILE_NAMEs="a b c d e"
fi

#for dset in T D E; do
    for name in ${FILE_NAMEs}; do    
        fname=flac_${dset}_a${name}.tar
        if [ ! -d ${ASVspoof5_dir}/flac_${dset} ]; then
            link="https://zenodo.org/records/14498691/files/${fname}?dowbload=1"
            echo -e "${RED}Downloading ${fname}"
            echo ${link}
            wget -q --show-progress -c ${link} -O  $fname
            tar -xvf ${fname}
            rm ${name}.tar
    fi
    done
#done
echo -e "${RED} We have ASVspoof5 database on "${ASVspoof5_dir}' now'

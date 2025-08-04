#!/bin/bash
# Lin Zhang created in July 12, 2025.
set -x
ITW_dir=$1

cd ${ITW_dir} # cd to /path/to/PartialSpoof

link="https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download"
if [ ! -d ${ITW_dir} ]; then
    echo -e "${RED}Downloading ITW"
    echo ${link}
    wget -q --show-progress -c ${link} -O  release_in_the_wild.zip
    unzip ${file}.zip
    rm ${file}.zip
fi
cd -
echo 'We have in_the_wild database on '${ITW_dir}' now'

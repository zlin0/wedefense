#!/bin/bash
set -x
PS_dir=$1
FILE_NAMEs="train dev eval protocols"

cd ${PS_dir}/../ # cd to /path/to/PartialSpoof
for file in ${FILE_NAMEs}; do

    link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
    if [ ! -d ./database/${file} ] && [ ! -d ./database/${file}/con_wav ]; then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	echo ${link}
        wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
        tar -zxvf database_${file}.tar.gz
        rm database_${file}.tar.gz
    fi
done
cd -
echo 'We have PartialSpoof database on '${PS_PATH}' now'

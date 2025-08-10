#!/bin/bash
set -ex
PS_dir=$1
FILE_NAMEs="train dev eval protocols"

# This script expects PS_dir to be /path/to/PartialSpoof/database
# It will change to /path/to/PartialSpoof to download and extract the data.
cd "${PS_dir}/../"

for file in ${FILE_NAMEs}; do
    link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
    # Only download if the target directory doesn't exist.
    if [ ! -d ./database/${file} ] && [ ! -d ./database/${file}/con_wav ]; then
        echo -e "Downloading PartialSpoof ${file}"
        echo "URL: ${link}"
        wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
        tar -zxvf database_${file}.tar.gz
        rm database_${file}.tar.gz
    fi
done
cd -  # Go back to original directory silently
echo "PartialSpoof database is ready at '${PS_dir}'"


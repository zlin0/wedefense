#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color

cache_dir=$HOME/.cache/huggingface/hub/datasets--jungjee--spoofceleb
spoofceleb_dir=/mnt/matylda4/qzhang/workspace/data/SpoofCeleb

if [ ! -d ${cache_dir} ]; then
    echo -e "Please get premission from author first"
    echo -e "Then, login huggingface-cli and download databases by yourself. "
    echo -e "Here are commands for your reference"
    echo -e "${RED} huggingface-cli login${NC}"
    echo -e "${RED} huggingface-cli download --repo-type dataset jungjee/spoofceleb --local-dir ${spoofceleb_dir}${NC}"
    echo -e "${RED} ${NC}"
fi

huggingface-cli download --repo-type dataset jungjee/spoofceleb --local-dir ${spoofceleb_dir}/backup
#
## Check whether spoofceleb is downloaded to local_dir or cache.
## If it is downloaded to cache, use the below script to move
if [[ $(hostname -f) == *clsp.jhu.edu   ]]; then
    find $HOME/.cache/huggingface/hub/datasets--jungjee--spoofceleb -type l \
        -exec ./move_to_here.sh ${spoofceleb_dir}/backup {} +

    echo "Merging tar parts into a single .tar.gz..."
    cat ${spoofceleb_dir}/backup/spoofceleb.tar.gz* > ${spoofceleb_dir}/backup/spoofceleb.tar.gz

    echo "Unpacking SpoofCeleb..."
    tar -zxvf ${spoofceleb_dir}/backup/spoofceleb.tar.gz -C ${spoofceleb_dir}/
fi


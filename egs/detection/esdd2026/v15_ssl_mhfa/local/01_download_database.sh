#!/bin/bash
set -ex
EnvSDD_dir=$1
#EnvSDD_dir=./
download_dir=${EnvSDD_dir}/download
#
cd ${EnvSDD_dir}/../ # cd to /path/to/PartialSpoof
if [ ! -d ${download_dir}  ]; then
  mkdir -p ${download_dir}
  (
  echo -e "${RED}Downloading EnvSDD-development"
  # Also the development data for track 1 from Zenodo.
  for i in {01..25}; do
      file=development.z${i}    
      link="https://zenodo.org/records/15220951/files/"${file}"?download=1"
      wget -q --show-progress -c ${link} -O  $download_dir/${file}
  done
  file=development.zip  
  link="https://zenodo.org/records/15220951/files/"${file}"?download=1"
  wget -q --show-progress -c ${link} -O $download_dir/${file}
  zip -s- $download_dir/development.zip -O $download_dir/dev_combined.zip
  unzip -d ${download_dir} ${download_dir}/dev_combined.zip
  ) &
  (
  echo -e "${RED}Downloading EnvSDD-test"
  for i in {01..05}; do
      file=test.z${i}   
      link="https://zenodo.org/records/15241138/files/"${file}"?download=1"
      wget -q --show-progress -c ${link} -O  $download_dir/${file} &
      echo ${link}
  done
  wait
  file=test.zip 
  link="https://zenodo.org/records/15241138/files/"${file}"?download=1"
  wget -q --show-progress -c ${link} -O  $download_dir/${file}
  zip -s- $download_dir/test.zip -O $download_dir/test_combined.zip
  unzip -d ${download_dir} ${download_dir}/test_combined.zip
  )&
  (
  # Track 2: 
  file="dev_track2_eval_track1_2.zip"
  link="https://zenodo.org/records/16684355/files/${file}?download=1"
  wget -q --show-progress -c ${link} -O  $download_dir/${file}
  unzip -d ${download_dir} ${download_dir}/${file}
  )&
  wait
fi

for i in {1..2}; do
  file=dev_track${i}.csv 
  link="https://raw.githubusercontent.com/apple-yinhan/EnvSDD/refs/heads/main/metadata/${file}"
  wget -q --show-progress -c ${link} -O  $download_dir/${file}
done

cd -
echo 'We have EnvSDD-Dev on '${EnvSDD_PATH}' now'

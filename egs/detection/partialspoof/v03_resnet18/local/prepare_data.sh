#!/bin/bash
#
#
# 1. download data
# 2. wav.scp
PS_dir=$1
data=$2

for dset in train dev eval; do
  if [ ! -d ${data}/${dset} ]; then
     mkdir -p ${data}/${dset}
  fi

  find ${PS_dir}/${dset}/con_wav -name "*.wav" | awk -F"/" '{print $NF,$0}' | sort >${data}/${dset}/wav.scp
  sed -i 's/\.wav / /g' ${data}/${dset}/wav.scp
  # check row number.

  # produce utt2cls from protocols
  cut -d' ' -f2,5 ${PS_dir}/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.${dset}.trl.txt > ${data}/${dset}/utt2cls

  ./tools/utt2spk_to_spk2utt.pl ${data}/${dset}/utt2cls >${data}/${dset}/cls2utt
done

echo "Prepared data folder for partialspoof, including wav.scp, utt2cls, cls2utt"


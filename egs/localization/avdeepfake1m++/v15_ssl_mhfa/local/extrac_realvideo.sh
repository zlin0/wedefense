egs/localization/avdeepfake1m++/#!/bin/bash

# 1. 

#2. extract real_video:
#2.1 real.mp4

# To avoid silence files
for dset in train val; do 
  # source_dir="./data/avdeepfake1mPP/${dset}"
  # save_dir="./data/avdeepfake1mPP/${dset}_audioonly_realvideo/"
  # mkdir -p "${save_dir}"

  # for x in train val; do 
  #   awk '{print $1, $6}' ./${source_dir}/wav.scp | sed 's/\.mp4$/.json/g' - > ${source_dir}/metadata.scp; 
  # done

  # 1. Filter metadata.json by modify_type and generate separate .scp files
  #     python ./local/filter_metadata_by_type.py   \
  #     --metadata_json /export/fs05/arts/dataset/AV-Deepfake1M-PlusPlus/${dset}_metadata.json   \
  #     --output_dir ./data/avdeepfake1mPP/${dset}

# ============================================================
# Summary:
#   audio_modified      : 266115 items
#   both_modified       : 265812 items
#   real                : 297389 items
#   visual_modified     : 269901 items
# ============================================================
# ============================================================
# Summary:
#   audio_modified      :  18938 items
#   both_modified       :  19069 items
#   real                :  20220 items
#   visual_modified     :  19099 items
# ============================================================

# 'both_modified', 'real', 'visual_modified', 'audio_modified'
# 2. create folder with real vedio, nosil

  source_dir="./data/avdeepfake1mPP/${dset}/"
  save_dir="./data/avdeepfake1mPP_nosil/${dset}_audioonly_realvideo/"
  mkdir -p "${save_dir}"

  # print some statistic info:
  echo "----Processing dataset: ${dset}----"
  total_items=$(wc -l < ${source_dir}/metadata.scp)
  echo "Total items in metadata.scp: ${total_items}"
  real_items=$(wc -l < ${source_dir}/metadata_real.scp)
  echo "Total real video items: ${real_items}"
  modified_items=$(wc -l < ${source_dir}/metadata_audio_modified.scp)
  echo "Total audio modified items: ${modified_items}"
  silent_items=$(grep -c "silent_videos" ${source_dir}/metadata.scp)
  echo "Total silent video items: ${silent_items}"

  # Combine real + audio_modified (with silent videos) then remove silent videos
  cat ${source_dir}/metadata_audio_modified.scp ${source_dir}/metadata_real.scp > ${save_dir}/tmp_withsil_metadata.scp
  silent_items=$(grep -c "silent_videos" ${save_dir}/tmp_withsil_metadata.scp)
  echo "Total silent video items (within real+audio_modified): ${silent_items}"
  grep -v "silent_videos" ${save_dir}/tmp_withsil_metadata.scp > ${save_dir}/metadata.scp

  new_items=$(wc -l < ${save_dir}/metadata.scp)
  echo "After removing silent videos, total items in metadata.scp: ${new_items}"


  for x in utt2dur utt2lab  wav.scp; do 
      rm -f ${x}; 
      awk -F' ' '(NR==FNR){UTT[$1]=$0}(NR!=FNR){if($1 in UTT){print}}
                ' ./${dset}_audioonly_realvideo/metadata.scp ../${dset}/${x} \
	      > ${dset}_audioonly_realvideo/${x} ; 
  done
  x=rttm
  awk -F' ' '(NR==FNR){UTT[$1]=$0}(NR!=FNR){if($2 in UTT){print}}
          ' ./${dset}_audioonly_realvideo/metadata.scp ../${dset}/${x} \
        > ${dset}_audioonly_realvideo/${x} ; 


done
# ----Processing dataset: train----
# Total items in metadata.scp: 1099217
# Total real video items: 297389
# Total audio modified items: 266115
# Total silent video items: 7583
# Total silent video items (within real+audio_modified): 3792
# After removing silent videos, total items in metadata.scp: 559712
# ----Processing dataset: val----
# Total items in metadata.scp: 77326
# Total real video items: 20220
# Total audio modified items: 18938
# Total silent video items: 398
# Total silent video items (within real+audio_modified): 199
# After removing silent videos, total items in metadata.scp: 38959
#label2id 



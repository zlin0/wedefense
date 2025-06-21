#!/bin/bash

# Copyright (c) 2025 Johan Rohdin (rohdin@fit.vutbr.cz)                   
#               
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# The DCF cost parameters used in the evaluation package
# Pspoof = 0.05 # Prior probability of a spoofing attack
# Cmiss  = 1    # Cost of CM system falsely rejecting target speaker
# Cfa    = 10   # Cost of CM system falsely accepting nontarget speaker

# Data statistics
# Dev:  #trials 24844, #tar 2548, #non 22296
# Test: #trials 71237, #tar 7355, #non 63882



#set -x
. ./path.sh || exit 1

stage=1
stop_stage=5

# Nothing will be written here so this path can be set to one of data dirs of
# the subsystems.
data='/mnt/matylda6/rohdin/software/wedefense/wedefense_20250602/egs/detection/asvspoof5/v15_ssl_mhfa/data/asvspoof5/'


# To ASVSPOOF
#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline/posteriors/dev/detection_result.txt
#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/dev/detection_result.txt

# Specify all systems here. Should match for dev. and test data


#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross
#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross

#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/
#/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp

# Since here we evaluate ASVspoof5f (A5), using EMBEDDING EXTRACTORS trained on partialspoof (PS) will result in cross-database fusion. Note that this refers to
# embedding extractor training data. THE CALIBRATION / FUSION MODELS ARE ALWAYS TRAINED ON INDOMAIN (A5) IN THESE EXPERIMENTS.

false && {
# Dev scores. Keys must be the systems defined above. 
# Dev scores. Keys must be the systems defined above. 
declare -A dev_scores=(
    ["PS_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v03_resnet18/exp/resnet.yaml/posteriors/dev/llr.txt"
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v13_ssl_aasist/exp/XLSR_AASIST_v1.yaml/posteriors/dev/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_xlsr53-FT-3_frozen/posteriors/dev/llr.txt'
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/cross-database/detection/partialspoof/v14_ssl_sls/exp/asvspoof5_v14_ssl_sls_SLS_w2v.yaml_4125427/posteriors/dev/llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/dev/llr.txt"
)
# Eval scores. Keys must be the systems defined above.
declare -A eval_scores=(
    ["PS_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v03_resnet18/exp/resnet.yaml/posteriors/eval/llr.txt"
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v13_ssl_aasist/exp/XLSR_AASIST_v1.yaml/posteriors/eval/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_xlsr53-FT-3_frozen/posteriors/eval/llr.txt'
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/cross-database/detection/partialspoof/v14_ssl_sls/exp/asvspoof5_v14_ssl_sls_SLS_w2v.yaml_4125427/posteriors/eval//llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/eval/llr.txt"
)

# Dev scores. Keys must be the systems defined above.
declare -A dev_scores=(
    ["PS_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v03_resnet18/exp/resnet.yaml/posteriors/dev/llr.txt"
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v13_ssl_aasist/exp/XLSR_AASIST_v1.yaml/posteriors/dev/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_xlsr53-FT-3_frozen/posteriors/dev/llr.txt'
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v14_ssl_sls/exp/SLS_w2v.yaml_4125427/posteriors/dev/llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/dev/llr.txt"
                      )


# Eval scores. Keys must be the systems defined above.
declare -A eval_scores=(
    ["PS_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v03_resnet18/exp/resnet.yaml/posteriors/eval/llr.txt"
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/partialspoof/v13_ssl_aasist/exp/XLSR_AASIST_v1.yaml/posteriors/eval/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_xlsr53-FT-3_frozen/posteriors/eval/llr.txt'
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v14_ssl_sls/exp/SLS_w2v.yaml_4125427/posteriors/eval/llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/partialspoof/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/eval/llr.txt"
)
##
}

declare -A dev_scores=(
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/cross-database/detection/asvspoof5/v13_ssl_aasist/exp/partialspoof_v13_ssl_aasist_XLSR_AASIST_v1.yaml/posteriors/dev/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/dev/llr.txt'
    ["A5_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v03_resnet18/exp/resnet.yaml_4112672/posteriors/dev/llr.txt"
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v14_ssl_sls/exp/SLS_w2v.yaml_4125427/posteriors/dev/llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline/posteriors/dev/llr.txt"
		      )


# Eval scores. Keys must be the systems defined above.
declare -A eval_scores=(
    ["PS_v12"]="----/llr.txt"
    ["PS_v13"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/cross-database/detection/asvspoof5/v13_ssl_aasist/exp/partialspoof_v13_ssl_aasist_XLSR_AASIST_v1.yaml/posteriors/eval/llr.txt"
    ["PS_v15"]='/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline_cross/posteriors/eval/llr.txt'
    ["A5_v03"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v03_resnet18/exp/resnet.yaml_4112672/posteriors/eval/llr.txt"
    ["A5_v12"]="----/llr.txt"
    ["A5_v14"]="/mnt/matylda6/rohdin/software/wedefense/scores_from_xin/detection/asvspoof5/v14_ssl_sls/exp/SLS_w2v.yaml_4125427/posteriors/eval/llr.txt"
    ["A5_v15"]="/mnt/matylda6/pengjy/shared_model_weights/lin/wedefense/egs/detection/asvspoof5/v15_ssl_mhfa/exp/MHFA_wav2vec2_large_960-FT-1stage_baseline/posteriors/eval/llr.txt"
)
###

# Specify the systems you want to test in current fusion here. Only these 
# systems will be used.
# For example use this if you want fuse the three systems
#declare -a systems=('v03_resnet18' 'MHFA_wavlm_large' 'MHFA_wavlm_plus')
# For example, use this if you only want calibrate v03_resnet18


#declare -a systems=('A5_v03')
#fusion_name=fusion-A5_v03

#declare -a systems=('A5_v14')
#fusion_name=fusion-A5_v14

declare -a systems=('A5_v15')
fusion_name=fusion-A5_v15

#declare -a systems=('PS_v13')
#fusion_name=fusion-PS_v13

#declare -a systems=('PS_v15')
#fusion_name=fusion-PS_v15

#declare -a systems=('A5_v14' 'A5_v15')
#fusion_name=fusion-A5_v14-A5_v15

#declare -a systems=('PS_v13' 'PS_v15')
#fusion_name=fusion-PS_v13-PS_v15

#declare -a systems=('PS_v15' 'A5_v15')
#fusion_name=fusion-PS_v15-A5_v15




# To match the operating point (OP) of the evaluation package, set
p_tar=0.95
c_fr=1
c_fa=10
# Neutral
#p_tar=0.5 
#c_fr=1 
#c_fa=1


exp_dir=exp/${fusion_name}_p_tar-${p_tar}_c_fr-${c_fr}_c_fa_${c_fa}
mkdir -p $exp_dir


# Stage 1. Copy scores to a convenient organization. (We also need the scores in
# our own folder since the evaluation script creates the results files in the 
# same directory as the input scores.) 
################################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    rm -r $exp_dir/input_scores/flac_D/
    for sys in ${systems[@]};do
	mkdir -p $exp_dir/input_scores/flac_D/$sys
        cp ${dev_scores[$sys]} $exp_dir/input_scores/flac_D/$sys/llr.txt
    done

    rm -r $exp_dir/input_scores/flac_E_eval/
    for sys in ${systems[@]};do
	mkdir -p $exp_dir/input_scores//flac_E_eval/$sys
        cp ${eval_scores[$sys]} $exp_dir/input_scores/flac_E_eval/$sys/llr.txt
    done
fi

	       
################################################################################
# Stage 2. Check performance of individual systems. 
################################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    for dset in flac_D flac_E_eval; do
	echo "Measuring Performance on $dset set..."
	for sys in ${systems[@]};do
	    echo $sys
	    # Preparing trials
	    # filename        cm-label
	    echo "filename cm-label" > ${data}/${dset}/cm_key_file.txt    
	    cat ${data}/${dset}/utt2cls >> ${data}/${dset}/cm_key_file.txt
	    sed -i "s/ /\t/g" ${data}/${dset}/cm_key_file.txt
	    echo $exp_dir/input_scores//$dset/$sys/llr.txt 
	    echo "Measuring " $dset
        python wedefense/metrics/detection/evaluation.py \
	    --m t1 \
	    --cm $exp_dir/input_scores//$dset/$sys/llr.txt \
	    --cm_key ${data}/${dset}/cm_key_file.txt
        done
    done
fi

		   
################################################################################
# Stage 3. Train logisic regression fusion / calibration model. 
################################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Estimate calibration parameters ..."
    # The target prior p_tar can be a single number in (0,1) a range in numpy 
    # arange  format, e.g., "np.arange(0.1,1,0.1)" in which case all these 
    # values will be tested and the best model will be saved.
    # The effective target priors p_tar will be converted to effective target 
    # prior according to
    #   p_tar_eff = p_tar * c_fr / (p_tar * c_fr + (1-p_tar) * c_fa) 
    python wedefense/postprocessing/train_calibration_fusion_model.py \
       --score_dir $exp_dir/input_scores/flac_D/ \
       --cm_key ${data}/flac_D/cm_key_file.txt  \
       --model_save_path ${exp_dir}/logistic_regression_fusion.pt \
       --p_tar $p_tar \
       --c_fr $c_fr \
       --c_fa $c_fa
fi

# However, this doesn't necessarily result in better performance. For a given
# evaluation OP, the best OP for training may, in practice, have to be tuned.
    

#######################################################################################
# Stage 4. Create fused scores 
#######################################################################################
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Compute fused scores ..."
    for dset in flac_D flac_E_eval; do
	mkdir -p $exp_dir/output_scores/$dset/
	python wedefense/postprocessing/apply_calibration_fusion_model.py \
               --model_load_path ${exp_dir}/logistic_regression_fusion.pt \
               --score_dir $exp_dir/input_scores/$dset/ \
               --new_scores_file $exp_dir/output_scores/$dset/llr.txt
    done
fi


#######################################################################################
# Stage 5. Measuring performance 
#######################################################################################
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for dset in flac_D flac_E_eval; do
	echo "Measuring " $dset
        python wedefense/metrics/detection/evaluation.py \
	    --m t1 \
	    --cm $exp_dir/output_scores/$dset/llr.txt \
	    --cm_key ${data}/${dset}/cm_key_file.txt
    done
fi






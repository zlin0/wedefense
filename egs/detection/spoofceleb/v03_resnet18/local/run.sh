
spoofceleb_dir=/mnt/matylda4/qzhang/workspace/data/SpoofCeleb

find $HOME/.cache/huggingface/hub/datasets--jungjee--spoofceleb -type l \
    -exec ./move_to_here.sh ${spoofceleb_dir} {} +


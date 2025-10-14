#!/bin/sh
set -e

dest_dir=$1
shift


if [ -z "$dest_dir" ]; then
    echo "Usage: $0 <destination_dir> <symlink1> [symlink2 ...]"
    exit 1
fi

mkdir -p $dest_dir

for link; do
    test -h "$link" || continue

    dir=$(dirname "$link")
    reltarget=$(readlink "$link")
    case $reltarget in
        /*) abstarget=$reltarget;;
        *)  abstarget=$dir/$reltarget;;
    esac
    filename=$(basename $link)

    rm -fv "$link"

    # Copy file to target directory
    cp -afv $abstarget ${dest_dir}/${filename} || {
        # on failure, restore the symlink
        rm -rfv $link
        ln -sfv $reltarget $link

    }
done

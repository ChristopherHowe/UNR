#!/bin/bash
set -e

# ln -s target symlinkLocation
location=$PWD
prefix_to_remove="$HOME/personal/UNR"

# Remove the specified prefix from the original path
new_path="${location#$prefix_to_remove}"
new_path="$(dirname $new_path)"

echo "Original Path: $location"
echo "New Path: $new_path"



target=/home/chris/personal$new_path
if [ ! -d "$target" ]; then
    echo Failed to create symlink directory $target does not exist.
    exit
fi

echo Please note you are about to create a symlink from link location:$target to destination:$location.
read -p "Press Enter to continue..."

ln -s $PWD $target

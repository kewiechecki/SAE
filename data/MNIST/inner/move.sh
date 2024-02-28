#!/bin/bash

# Function to move files to their respective directories
move_files() {
    local prefix=$1
    local dir_name=$2
    local path=$3

    # Create the directory if it doesn't exist
    mkdir -p "$path/$dir_name"

    # Find and move files
    find "$path" -maxdepth 1 -type f -name "${prefix}*" | while read -r file; do
        new_name="${file#$path/$prefix}"
        mv "$file" "$path/$dir_name/$new_name"
    done
}

# Recursively walk through directories
find . -type d | while read -r dir; do
    move_files "L1_L2" "L1_L2" "$dir"
    move_files "L2" "L2" "$dir"
done

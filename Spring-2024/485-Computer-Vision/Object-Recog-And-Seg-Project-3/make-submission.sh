#!/bin/bash
set -e

SUB_DIR=Project3
ZIP_NAME=howe_christopher_project3.zip

if [ -d $SUB_DIR ]; then
    rm $SUB_DIR -r
fi
mkdir $SUB_DIR

cp README.md ./$SUB_DIR/README.txt
cp project3.py ./$SUB_DIR

zip $ZIP_NAME $SUB_DIR -r


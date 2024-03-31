#!/bin/bash
set -e

SUB_DIR=Project2
ZIP_NAME=howe_christopher_project2.zip

mkdir $SUB_DIR

cp README.md ./$SUB_DIR/README.txt
cp project2.py ./$SUB_DIR

zip $ZIP_NAME $SUB_DIR  


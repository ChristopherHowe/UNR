#!/bin/bash
set -e

SUB_DIR=PA3_Howe_Christopher
ZIP_DIR=$SUB_DIR.zip

mkdir  $SUB_DIR

cp makefile $SUB_DIR/Makefile
cp sched.c $SUB_DIR
cp ./latex-out/main.pdf ./$SUB_DIR/PA3_Howe_Christopher_Questions.pdf

zip $ZIP_NAME $SUB_DIR

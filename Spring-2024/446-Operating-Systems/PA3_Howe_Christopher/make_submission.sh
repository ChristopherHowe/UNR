#!/bin/bash
set -e

SUB_DIR=PA3_Howe_Christopher
ZIP_NAME=$SUB_DIR.zip

mkdir  $SUB_DIR

cp makefile $SUB_DIR/Makefile
cp sched.c print_progress.c $SUB_DIR
cp ./Free-Response-Doc/.latex-out/main.pdf ./$SUB_DIR/PA3_Howe_Christopher_Questions.pdf


tar -cvzf $SUB_DIR.tar.gz $SUB_DIR

rm $SUB_DIR -r

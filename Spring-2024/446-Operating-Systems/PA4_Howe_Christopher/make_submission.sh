#!/bin/bash
set -e

SUB_DIR=PA4_Howe_Christopher

mkdir  $SUB_DIR


cp .latex-out/main.pdf ./$SUB_DIR/PA4_Howe_Christopher_Questions.pdf


tar -cvzf $SUB_DIR.tar.gz $SUB_DIR

rm $SUB_DIR -r

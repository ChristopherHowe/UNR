#!/bin/bash
set -e

SUB_DIR=PA5_Howe_Christopher

mkdir  $SUB_DIR

cp makefile $SUB_DIR/Makefile
cp myfs.c $SUB_DIR

tar -cvzf $SUB_DIR.tar.gz $SUB_DIR

rm $SUB_DIR -r

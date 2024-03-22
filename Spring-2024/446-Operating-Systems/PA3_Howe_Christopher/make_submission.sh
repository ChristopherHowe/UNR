#!/bin/bash
set -e

SUB_DIR=PA3_Howe_Christopher
ZIP_DIR=$SUB_DIR.zip

mkdir  $SUB_DIR
cp makefile $SUB_DIR/Makefile
cp sched.c $SUB_DIR

zip $ZIP_NAME $SUB_DIR

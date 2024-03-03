#!/bin/bash
set -e

pa_name=PA2_Howe_Christopher

mkdir $pa_name
cd $pa_name
mkdir loop
mkdir thread
cp ../loop/looped_sum.c ../loop/makefile ./loop
cp ../thread/threaded_sum.c ../thread/makefile ./thread
cp ../Free-Response-Doc/.latex-out/main.pdf ./$pa_name.pdf
cd ..

# Create the tarball
tar -cvzf $pa_name.tar.gz $pa_name

# delete the created folder
rm -r $pa_name




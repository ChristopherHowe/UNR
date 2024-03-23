#!/bin/bash
set -e

cd $PWD
make
./sched 4

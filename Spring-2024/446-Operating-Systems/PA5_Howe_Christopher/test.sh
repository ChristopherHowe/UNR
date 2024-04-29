#!/bin/bash
set -e

EXECUTABLE_NAME=myfs
cd $PWD
make 

info(){
    teal_color='\e[1;36m'
    white_color='\e[0m'
    echo -e "${teal_color}$@${white_color}"
}

if [ -z "$1" ]; then
    echo please specify a problem demonstration to run.
    echo Problem 1:
    exit
fi 

if [ "$1" -eq 1 ]; then
    info Executing test for problem 1 \(./myfs\)
    ./myfs
    exit
fi

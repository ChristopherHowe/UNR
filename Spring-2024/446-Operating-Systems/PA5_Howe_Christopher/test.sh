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

info Executing myfs executable
./myfs

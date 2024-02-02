#!/bin/bash
set -e

numProblems=1
run_problem(){
    prob_num=$1
    if [ $prob_num -eq 1 ]; then
        g++ -o arrayReverse problem1.cpp
        ./arrayReverse
        rm arrayReverse
    fi
}

if [ $1 -eq 0 ]; then
    for ((i = 1; i <= $numProblems; i++)); do
        run_problem $i
    done
else 
    run_problem $1
fi
